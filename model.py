"""测试（quiz）的数据模型、校验与规范化。

一个测试 = 3 条二元轴（共 8 种结果）+ 6-10 道题 + 8 条结果文案。
每条选项用 ``scores`` 在轴上加分（-1/0/+1），最后按各轴符号拼成 3 位二进制 key。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AXIS_COUNT = 3
RESULT_KEYS = ["".join(bits) for bits in __import__("itertools").product("01", repeat=AXIS_COUNT)]
RESULT_FIELDS = ["name", "emoji", "tags", "desc", "strength", "weak", "match"]
MIN_QUESTIONS, MAX_QUESTIONS = 6, 10
OPTIONS_PER_QUESTION = 3


class QuizError(ValueError):
    """数据不合法。"""


@dataclass
class Axis:
    key: str
    neg: str
    pos: str

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "neg": self.neg, "pos": self.pos}


def quality_problems(quiz: dict[str, Any]) -> list[str]:
    """在结构校验之外，检查"生成内容像不像人写的"：重复、失衡、emoji 不对。

    这些问题不阻塞渲染，但会触发模型重试；因为「8 种结果里有一种占 50%」这种内容
    上线就是翻车，必须在生成阶段拦下来。
    """
    problems: list[str] = []
    results = quiz.get("results") or {}
    names = [result.get("name", "") for result in results.values()]
    if len(set(names)) < len(names):
        problems.append("8 个结果的名称有重复，需要两两不同")

    bad_emoji = [key for key, result in results.items() if not _is_emoji(result.get("emoji", ""))]
    if bad_emoji:
        problems.append(f"结果的 emoji 不是单个 emoji（{', '.join(bad_emoji[:3])}），不能写汉字或拼多个 emoji")

    dup_emoji: dict[str, list[str]] = {}
    for key, result in results.items():
        dup_emoji.setdefault(str(result.get("emoji", "")), []).append(key)
    repeated = [emoji for emoji, keys in dup_emoji.items() if emoji and len(keys) > 1]
    if repeated:
        problems.append(
            f"结果 emoji 有重复（{', '.join(repeated[:3])}），8 个结果各用一个不同的 emoji")

    leaked = [key for key, result in results.items()
              if any(mark in str(result.get(field, "")) for field in ("name", "desc", "strength", "weak")
                     for mark in ("〈", "〉", "……", "xxx"))]
    if leaked:
        problems.append("文案里残留了模板占位符（〈〉/……），需要换成真实内容")

    generic = [key for key, result in results.items()
               if not result.get("match") or "同样" in str(result.get("match", ""))]
    if len(generic) >= 4:
        problems.append("match（合拍类型）文案太模板化，需要具体写出不同类型的名字")

    for axis in quiz.get("axes") or []:
        key = axis["key"]
        plus = minus = 0
        for question in quiz.get("questions") or []:
            for option in question.get("options") or []:
                value = (option.get("scores") or {}).get(key, 0)
                plus += 1 if value > 0 else 0
                minus += 1 if value < 0 else 0
        if min(plus, minus) < 2 or abs(plus - minus) > 4:
            problems.append(
                f"轴 {key} 的 +1/-1 分布失衡（+{plus}/-{minus}），每条轴的正反两侧都要出现 3 次以上"
            )

    shares = rarity(quiz)
    if shares:
        worst = max(shares.items(), key=lambda item: item[1])
        rarest = min(shares.items(), key=lambda item: item[1])
        if worst[1] > 22:
            problems.append(
                f"结果 {worst[0]}（{results[worst[0]]['name']}）占比 {worst[1]}% 过高，"
                "请让每个选项只打 1 条轴的 ±1，避免叠加"
            )
        if rarest[1] < 2:
            problems.append(f"结果 {rarest[0]} 几乎测不到（{rarest[1]}%），说明有轴被答死了")
    return problems


def _is_emoji(text: str) -> bool:
    """必须是"一个" emoji：汉字不行，把两个 emoji 拼一起也不行（ZWJ 组合除外）。"""
    text = (text or "").strip()
    if not text or len(text) > 6:
        return False
    bases = [
        char for char in text
        if ord(char) >= 0x1F000 or 0x2600 <= ord(char) <= 0x27BF or 0x2B00 <= ord(char) <= 0x2BFF
    ]
    if not bases:
        return False
    if "\u200d" in text:  # 🐈‍⬛ 这类 ZWJ 组合允许
        return len(bases) <= 3
    return len(bases) == 1


def _all_scores(quiz: dict[str, Any]):
    """穷举所有作答，产出 (每个轴的分数, 每个轴最后一个非零增量)。"""
    import itertools

    axes = [axis["key"] for axis in quiz.get("axes") or []]
    questions = quiz.get("questions") or []
    for combo in itertools.product(range(OPTIONS_PER_QUESTION), repeat=len(questions)):
        scores = {key: 0 for key in axes}
        last_delta = {key: 0 for key in axes}
        for question, choice in zip(questions, combo):
            options = question.get("options") or []
            if choice >= len(options):
                continue
            for axis_key, delta in (options[choice].get("scores") or {}).items():
                if axis_key in scores:
                    value = int(delta)
                    scores[axis_key] += value
                    if value:
                        last_delta[axis_key] = value
        yield scores, last_delta


def calibrate(quiz: dict[str, Any]) -> dict[str, Any]:
    """自动校准每条轴的判定阈值，让 8 种结果尽可能等概率。

    模型生成的题目经常出现"某条轴 80% 都为正"的情况，直接按 0 切分会让
    某种结果占 40%+。这里穷举所有作答，给每条轴找一个"刚好把人群劈成两半"的
    阈值（整数分，平局按该轴最后一个选择决定），并把阈值写回数据。
    """
    if not quiz.get("axes") or not quiz.get("questions"):
        return quiz
    axes = [axis["key"] for axis in quiz["axes"]]
    combos = list(_all_scores(quiz))
    total = len(combos)
    if not total:
        return quiz

    thresholds: dict[str, float] = {}
    for axis in axes:
        values = sorted({scores[axis] for scores, _ in combos})
        candidates = [value - 0.5 for value in values]  # 阈值取 x.5，score>t 即 score>=value
        candidates = [value for value in values] + [values[0] - 1, values[-1]]
        best_threshold, best_gap = 0, total + 1
        for threshold in candidates:
            ones = 0
            for scores, last_delta in combos:
                score = scores[axis]
                if score > threshold:
                    ones += 1
                elif score == threshold and last_delta[axis] > 0:
                    ones += 1
            gap = abs(ones - total / 2)
            if gap < best_gap:
                best_threshold, best_gap = threshold, gap
        thresholds[axis] = best_threshold

    calibrated = dict(quiz)
    calibrated["thresholds"] = thresholds
    return calibrated


def rarity(quiz: dict[str, Any]) -> dict[str, float]:
    """穷举所有作答组合，算出每种结果被抽到的百分比（用于结果页展示"稀有度"）。"""
    import itertools

    axes = [axis["key"] for axis in quiz.get("axes") or []]
    questions = quiz.get("questions") or []
    results = quiz.get("results") or {}
    if not axes or not questions:
        return {}
    thresholds = quiz.get("thresholds") or {}
    hits = {key: 0 for key in results}
    total = 0
    for scores, last_delta in _all_scores(quiz):
        total += 1
        bits = []
        for axis in axes:
            score = scores[axis]
            threshold = thresholds.get(axis, -0.5)
            if score > threshold:
                bits.append("1")
            elif score == threshold and last_delta[axis] > 0:
                bits.append("1")
            else:
                bits.append("0")
        key = "".join(bits)
        if key in hits:
            hits[key] += 1
    return {key: round(count / total * 100, 1) for key, count in hits.items()}


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(quiz: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(quiz, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def normalize(data: dict[str, Any], *, slug: str | None = None) -> dict[str, Any]:
    """把模型输出整理成合法结构；缺字段的尽量补，补不了的抛 QuizError。"""
    if not isinstance(data, dict):
        raise QuizError("顶层不是对象")

    quiz = {
        "slug": _slug(slug or data.get("slug") or data.get("title") or "quiz"),
        "title": _text(data.get("title"), "标题"),
        "subtitle": _text(data.get("subtitle", ""), "副标题", required=False),
        "emoji": _emoji(data.get("emoji"), "🎯"),
        "est_minutes": int(data.get("est_minutes") or 2),
        "share_hint": _text(data.get("share_hint", ""), "分享语", required=False),
        "axes": _axes(data.get("axes")),
        "questions": _questions(data.get("questions")),
        "results": _results(data.get("results")),
        "cta": data.get("cta") or "我也想要一个专属测试",
    }
    if not quiz["subtitle"]:
        quiz["subtitle"] = f"{len(quiz['questions'])} 道题，{quiz['est_minutes']} 分钟，看看你是哪一种"
    if not quiz["share_hint"]:
        quiz["share_hint"] = "快来看看我是哪一种，测完把你的结果发我"
    return quiz


def validate(quiz: dict[str, Any]) -> list[str]:
    """返回问题列表（空表示通过）。"""
    problems: list[str] = []
    axes = quiz.get("axes") or []
    if len(axes) != AXIS_COUNT:
        problems.append(f"轴数量应为 {AXIS_COUNT}，实际 {len(axes)}")
    keys = [a.get("key") for a in axes]
    if len(set(keys)) != len(keys):
        problems.append("轴 key 重复")
    questions = quiz.get("questions") or []
    if not MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS:
        problems.append(f"题目数量应在 {MIN_QUESTIONS}-{MAX_QUESTIONS}，实际 {len(questions)}")
    for i, question in enumerate(questions, 1):
        options = question.get("options") or []
        if len(options) != OPTIONS_PER_QUESTION:
            problems.append(f"第 {i} 题应有 {OPTIONS_PER_QUESTION} 个选项，实际 {len(options)}")
        for option in options:
            if not option.get("text"):
                problems.append(f"第 {i} 题有空选项")
            for axis_key in (option.get("scores") or {}):
                if axis_key not in keys:
                    problems.append(f"第 {i} 题的选项用了未知轴 {axis_key}")
    results = quiz.get("results") or {}
    for key in RESULT_KEYS:
        result = results.get(key)
        if not result:
            problems.append(f"缺少结果类型 {key}")
            continue
        for field in RESULT_FIELDS:
            if not result.get(field):
                problems.append(f"结果 {key} 缺少字段 {field}")
        if not isinstance(result.get("tags"), list) or len(result["tags"]) < 2:
            problems.append(f"结果 {key} 的 tags 至少 2 个")
    return problems


def _axes(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or len(raw) != AXIS_COUNT:
        raise QuizError(f"axes 必须是 {AXIS_COUNT} 条")
    axes: list[dict[str, str]] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise QuizError(f"第 {i} 条轴不是对象")
        axes.append({
            "key": _slug(item.get("key") or f"axis{i}"),
            "neg": _text(item.get("neg"), f"轴 {i} 的负向描述"),
            "pos": _text(item.get("pos"), f"轴 {i} 的正向描述"),
        })
    if len({a["key"] for a in axes}) != AXIS_COUNT:
        raise QuizError("轴 key 必须互不相同")
    return axes


def _questions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise QuizError("questions 必须是数组")
    questions: list[dict[str, Any]] = []
    for i, item in enumerate(raw[:MAX_QUESTIONS], 1):
        if not isinstance(item, dict) or not item.get("q"):
            continue
        options = []
        for option in (item.get("options") or [])[:OPTIONS_PER_QUESTION]:
            if not isinstance(option, dict) or not option.get("text"):
                continue
            scores = {
                _slug(key): _int(value, -1, 1)
                for key, value in (option.get("scores") or {}).items()
                if _int(value, -1, 1) != 0
            }
            options.append({"text": str(option["text"]).strip(), "scores": scores})
        if len(options) == OPTIONS_PER_QUESTION:
            questions.append({"q": str(item["q"]).strip(), "options": options})
    if len(questions) < MIN_QUESTIONS:
        raise QuizError(f"有效题目只有 {len(questions)} 道（至少 {MIN_QUESTIONS}）")
    return questions


def _results(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise QuizError("results 必须是对象（8 个 key）")
    results: dict[str, dict[str, Any]] = {}
    for key in RESULT_KEYS:
        item = raw.get(key) or raw.get(list(raw)[RESULT_KEYS.index(key)] if False else key)
        if not isinstance(item, dict):
            raise QuizError(f"缺少结果类型 {key}")
        tags = [str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()]
        results[key] = {
            "name": _text(item.get("name"), f"结果 {key} 的名称"),
            "emoji": _emoji(item.get("emoji"), "✨"),
            "tags": tags[:4] or ["神秘", "有趣"],
            "desc": _text(item.get("desc"), f"结果 {key} 的描述"),
            "strength": _text(item.get("strength", ""), "", required=False),
            "weak": _text(item.get("weak", ""), "", required=False),
            "match": _text(item.get("match", ""), "", required=False),
        }
        if not results[key]["strength"]:
            results[key]["strength"] = "这个类型有它自己的节奏。"
        if not results[key]["weak"]:
            results[key]["weak"] = "偶尔会有点小纠结。"
        if not results[key]["match"]:
            results[key]["match"] = "互补的类型最合拍。"
    return results


def _text(value: Any, label: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise QuizError(f"{label}不能为空")
    return text


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "")).strip("-").lower()
    return text or "quiz"


def _emoji(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if len(text) <= 4:  # emoji 或 ZWJ 序列
        return text
    return fallback


def _int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(low, min(high, number))
