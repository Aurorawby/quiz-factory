"""Quiz Factory 生成管线：骨架 → 题目 → 文案，三段式生成 + 每段独立门禁重试。

为什么分三段（v2 的核心改动）：
  单次生成整份 JSON 时，7B 模型的输出长度越长，类型名重复、占位符残留、
  轴分布失衡的概率越高（v1 实测 4 次只有 1 次整体过门禁）。
  分段后每段输出更短、任务更单一，一段不合格只重做那一段。

用法：
  python generate.py --theme "测测你是哪种打工人" --slug worker
  python generate.py --theme "..." --provider openai --model deepseek-chat
  python generate.py --gallery          # 只重建测试库索引
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import model
import presets
import render
from llm import LLMError, load_env_keys, make_client

ROOT = Path(__file__).resolve().parent
QUIZ_DIR = ROOT / "quizzes"
DIST_DIR = ROOT / "docs"
MANIFEST = DIST_DIR / "quizzes.json"

SYSTEM = (
    "你是资深互动内容策划，擅长设计「测测你是哪种 X」这类轻量人格测试。"
    "你只输出一个 JSON 对象，不写解释、不用 Markdown 代码块。"
)

STAGE_RETRIES = 2  # 每段最多重试次数（即最多 3 次尝试）

SkeletonCB = Callable[[str, str], None]


def _say(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)


def _default_progress(stage: str, msg: str) -> None:
    _say(f"  [{stage}] {msg}")


# ---------------------------------------------------------------- 各段提示词

SKELETON_PROMPT = """请为一个测试设计「骨架」，主题：{theme}

只做三件事，**不要写题目、不要写结果描述文案**：
1. title：给测试起个标题（默认「测测你是哪种{short_theme}」，可以更有梗）；
2. axes：恰好 3 条二元倾向轴。key 用英文小写（如 energy / social / order），
   neg / pos 各写一句 6-12 字的**中性**口语描述（不要价值判断，两边都要像普通人）；
3. results：恰好 8 个结果类型，key 依次为 000/001/010/011/100/101/110/111
   （对应 axes 顺序的二进制）。每个结果只有三个字段：
   - name：「两个字前缀 · 三个字特点」的格式（如「橘猫 · 摆烂大师」），**两两不同**，要贴合主题；
   - emoji：**单个** emoji 字符（不能是汉字、不能拼两个）；
   - tags：3 个短词。

只输出 JSON：
{{
  "title": "...",
  "emoji": "🎯",
  "subtitle": "一句话引导语（**不要写具体题目数量和时长**，系统会自动补）",
  "share_hint": "一句让用户分享结果的话",
  "axes": [{{"key": "energy", "neg": "...", "pos": "..."}}, 共3条],
  "results": {{"000": {{"name": "...", "emoji": "🐈", "tags": ["...","...","..."]}}, 共8个}}
}}"""

QUESTIONS_PROMPT = """测试主题：{theme}
测试的 3 条轴（考生在这三条倾向上的选择决定结果）：
{axes_text}

请设计恰好 {count} 道**具体生活场景题**（不要"你是否…"这种是非题）。
为了保证结果分布公平，题目结构必须严格遵守下面的规则：
1. 把 {count} 道题尽量均匀分配给 3 条轴（每条轴**至少 2 道**）；
2. 每道题**只考 1 条轴**，恰好 3 个选项：
   - 一个该轴**正向**选项（scores 里该轴 = 1）；
   - 一个该轴**负向**选项（scores 里该轴 = -1）；
   - 一个**中性**选项（scores 为空 {{}}）；
   三个选项的先后顺序随机打乱，不要固定正向在前；
3. 每个选项像一句真实的人话（12-25 字，有画面感），正向/负向都写成普通人的选择，不贬低任何一边；
4. 场景要具体、贴近主题，题目之间不重复套路。

只输出 JSON：
{{"questions": [{{"q": "〈场景题〉", "options": [
    {{"text": "〈正向选项〉", "scores": {{"energy": 1}}}},
    {{"text": "〈中性选项〉", "scores": {{}}}},
    {{"text": "〈负向选项〉", "scores": {{"energy": -1}}}}
]}}, ...]}}"""

COPY_PROMPT = """测试主题：{theme}
3 条轴：{axes_text}
8 个结果类型（key 是三条轴的二进制组合，下面已给出每个 key 的准确含义，文案必须贴合含义）：
{results_text}

请为每个结果逐条写文案，要求：
1. desc：60-90 字，第二人称「你」，有画面感，贴合这个类型的名字和上面给出的含义；
2. strength：一句「高光」（这个类型最闪的点）；
3. weak：一句「小毛病」，要善意，不贬低人；
4. match：合拍类型，**必须写另一个结果的具体名字**（不要写"同样的人"）；
5. 8 条文案风格一致但内容不重复；不刻板印象、不冒犯、不出现真实品牌/明星。

只输出 JSON：
{{"results": {{"000": {{"desc": "...", "strength": "...", "weak": "...", "match": "〈另一个结果名〉"}}, 共8个}}}}"""


# ---------------------------------------------------------------- 各段校验

def _check_skeleton(data: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    axes = data.get("axes")
    if not isinstance(axes, list) or len(axes) != 3:
        return ["axes 必须是 3 条"]
    keys = [str(a.get("key", "")).strip() for a in axes]
    if len(set(keys)) != 3 or not all(keys):
        problems.append("轴 key 重复或为空")
    for a in axes:
        for side in ("neg", "pos"):
            text = str(a.get(side, "")).strip()
            if not text:
                problems.append(f"轴 {a.get('key')} 的 {side} 描述为空")
    results = data.get("results")
    if not isinstance(results, dict) or len(results) != 8:
        return problems + ["results 必须是 8 个（000-111）"]
    names = [str((results.get(k) or {}).get("name", "")).strip() for k in _result_keys(results)]
    if len(set(names)) != 8 or any(not n for n in names):
        problems.append("8 个结果的 name 必须两两不同且非空")
    for key in _result_keys(results):
        item = results.get(key) or {}
        if not model._is_emoji(item.get("emoji", "")):
            problems.append(f"结果 {key} 的 emoji 不是单个 emoji")
        tags = item.get("tags")
        if not isinstance(tags, list) or len([t for t in tags if str(t).strip()]) < 2:
            problems.append(f"结果 {key} 的 tags 至少 2 个")
    for field in ("title",):
        if not str(data.get(field, "")).strip():
            problems.append("缺少 title")
    return problems


def _check_questions(axes: list[dict[str, str]], questions: list[dict[str, Any]],
                     count: int) -> list[str]:
    """v2.1 结构规则：每题只考 1 条轴（+1 / -1 / 中性各一个选项），平衡由结构保证。"""
    problems: list[str] = []
    keys = [a["key"] for a in axes]
    if len(questions) != count:
        problems.append(f"题目数量应为 {count}，实际 {len(questions)}")
    per_axis: dict[str, int] = {key: 0 for key in keys}
    for i, q in enumerate(questions, 1):
        options = q.get("options") or []
        if len(options) != 3:
            problems.append(f"第 {i} 题应有 3 个选项，实际 {len(options)}")
            continue
        scored_axes: set[str] = set()
        plus = minus = 0
        for opt in options:
            if not str(opt.get("text", "")).strip():
                problems.append(f"第 {i} 题有空选项")
            scores = opt.get("scores") or {}
            if len(scores) > 1:
                problems.append(f"第 {i} 题的选项给多条轴打了分（只能 1 条）")
            for key, value in scores.items():
                if key not in keys:
                    problems.append(f"第 {i} 题用了未知轴 {key}")
                    continue
                scored_axes.add(key)
                if int(value) > 0:
                    plus += 1
                elif int(value) < 0:
                    minus += 1
        if len(scored_axes) != 1:
            problems.append(f"第 {i} 题必须只考 1 条轴（现在考了 {len(scored_axes)} 条）")
            continue
        axis = next(iter(scored_axes))
        if plus != 1 or minus != 1:
            problems.append(
                f"第 {i} 题考轴 {axis} 时必须正好 1 个 +1、1 个 -1（第 3 个选项留中性）")
            continue
        per_axis[axis] += 1
    for key, n in per_axis.items():
        if n < 2:
            problems.append(f"轴 {key} 只有 {n} 道题（每条轴至少 2 道），请重新分配")
    return problems


def _check_copy(names: dict[str, str], copy: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    all_names = list(names.values())
    for key in _result_keys(names):
        item = copy.get(key) or {}
        for field in ("desc", "strength", "weak"):
            text = str(item.get(field, "")).strip()
            if not text:
                problems.append(f"结果 {key} 缺少 {field}")
            elif "〈" in text or "〉" in text or "xxx" in text.lower():
                problems.append(f"结果 {key} 的 {field} 残留模板占位符")
        match = str(item.get("match", "")).strip()
        if not match:
            problems.append(f"结果 {key} 缺少 match")
        elif not any(match != names[key] and match in n for n in all_names):
            problems.append(f"结果 {key} 的 match 必须写另一个结果的具体名字")
    return problems


def _result_keys(mapping: dict[str, Any]) -> list[str]:
    order = ["".join(bits) for bits in __import__("itertools").product("01", repeat=3)]
    present = set(mapping)
    return [key for key in order if key in present] or sorted(present)


def _feedback(prompt: str, problems: list[str]) -> str:
    return (prompt + "\n\n你上一版输出有以下问题，请修正后重新输出**完整** JSON：\n- "
            + "\n- ".join(problems[:8]))


# ---------------------------------------------------------------- 三段式管线

def _run_stage(client, stage: str, system: str, prompt: str, check: Callable[[dict[str, Any]], list[str]],
               *, temperature: float, max_tokens: int, report: dict[str, Any],
               progress: SkeletonCB) -> dict[str, Any]:
    """跑一段：生成 → 校验 → 不合格带反馈重试。全部失败时返回最后一份输出。"""
    attempts = []
    problems: list[str] = []
    data: dict[str, Any] = {}
    for attempt in range(STAGE_RETRIES + 1):
        progress(stage, f"第 {attempt + 1} 次尝试...")
        try:
            data = client.chat_json(system, prompt, temperature=temperature, max_tokens=max_tokens)
        except LLMError as exc:
            attempts.append({"attempt": attempt + 1, "ok": False, "error": str(exc)})
            progress(stage, f"调用失败：{exc}")
            continue
        problems = check(data)
        ok = not problems
        attempts.append({"attempt": attempt + 1, "ok": ok,
                         "problems": problems if not ok else []})
        if ok:
            progress(stage, "✓ 通过门禁")
            break
        progress(stage, f"门禁发现 {len(problems)} 个问题，带反馈重试")
        prompt = _feedback(prompt, problems)
    stage_report = {"stage": stage, "attempts": attempts,
                    "ok": bool(attempts and attempts[-1]["ok"]),
                    "tries": len(attempts)}
    report["stages"].append(stage_report)
    return data


def generate(theme: str, slug: str, *, provider: str = "ollama", model_name: str = "qwen2.5:7b",
             base_url: str = "", api_key: str = "", count: int = 8,
             progress: SkeletonCB = _default_progress) -> tuple[dict[str, Any], dict[str, Any]]:
    """三段式生成一个测试，返回 (quiz, report)。report 记录每段尝试与最终问题。"""
    started = time.perf_counter()
    report: dict[str, Any] = {"theme": theme, "provider": provider, "model": model_name,
                              "slug": slug, "stages": [], "first_pass": False,
                              "final_problems": [], "needs_polish": True}
    env = load_env_keys()
    client = make_client(provider, model=model_name, base_url=base_url,
                         api_key=api_key or env.get("QUIZ_API_KEY") or env.get("LIT_API_KEY", ""))
    if not base_url and provider != "ollama":
        base_url = env.get("QUIZ_API_BASE") or env.get("LIT_API_BASE", "")
        if base_url:
            client.base_url = base_url.rstrip("/")
    short_theme = theme.replace("测测你是哪种", "").replace("测测你是哪个", "").strip()
    progress("skeleton", f"开始：{theme}（{provider}/{model_name}）")

    # ---- 第 1 段：骨架（轴 + 8 个类型名） ----
    skel_prompt = SKELETON_PROMPT.format(theme=theme, short_theme=short_theme)
    skeleton = _run_stage(client, "skeleton", SYSTEM, skel_prompt, _check_skeleton,
                          temperature=0.7, max_tokens=1400, report=report, progress=progress)
    skeleton.update({"slug": slug, "est_minutes": 2})
    axes = skeleton.get("axes") or []

    # ---- 第 2 段：题目 ----
    axes_text = "\n".join(f"- {a['key']}：负向「{a['neg']}」←→ 正向「{a['pos']}」" for a in axes)
    q_prompt = QUESTIONS_PROMPT.format(theme=theme, axes_text=axes_text, count=count)

    def _check_q(data: dict[str, Any]) -> list[str]:
        questions = data.get("questions")
        if not isinstance(questions, list):
            return ["questions 必须是数组"]
        return _check_questions(axes, questions, count)

    q_data = _run_stage(client, "questions", SYSTEM, q_prompt, _check_q,
                        temperature=0.65, max_tokens=2600, report=report, progress=progress)
    questions = q_data.get("questions") or []

    # ---- 第 3 段：文案 ----
    skel_results = skeleton.get("results") or {}
    names = {key: str((skel_results.get(key) or {}).get("name", "")) for key in _result_keys(skel_results)}

    def _key_meaning(key: str) -> str:
        bits = list(key)
        parts = []
        for i, axis in enumerate(axes):
            parts.append(f"{axis['key']}={'正向' if i < len(bits) and bits[i] == '1' else '负向'}")
        return "，".join(parts)

    results_text = "\n".join(
        f"- {key}：{name}（{', '.join((skel_results.get(key) or {}).get('tags', []))}）"
        f"  含义：{_key_meaning(key)}"
        for key, name in names.items())
    c_prompt = COPY_PROMPT.format(theme=theme, axes_text=axes_text, results_text=results_text)
    copy_data = _run_stage(client, "copy", SYSTEM, c_prompt,
                           lambda d: _check_copy(names, d.get("results") if isinstance(d.get("results"), dict) else {}),
                           temperature=0.7, max_tokens=2400, report=report, progress=progress)

    # ---- 组装 + 校准 + 总门禁 ----
    raw_results: dict[str, dict[str, Any]] = {}
    copy_results = copy_data.get("results") if isinstance(copy_data.get("results"), dict) else {}
    for key in _result_keys(skel_results):
        base = dict(skel_results.get(key) or {})
        base.update({f: (copy_results.get(key) or {}).get(f) for f in ("desc", "strength", "weak", "match")})
        raw_results[key] = base
    raw = {**skeleton, "axes": axes, "questions": questions, "results": raw_results}
    quiz = model.normalize(raw, slug=slug)
    # 引导语里的数字（题数/时长）必须由代码生成，不信模型（7B 实测会写"30道题"）
    if any(ch.isdigit() for ch in quiz["subtitle"]):
        quiz["subtitle"] = f"{len(quiz['questions'])} 道题，{quiz['est_minutes']} 分钟，看看你是哪一种"
    quiz = model.calibrate(quiz)
    final_problems = model.validate(quiz) + model.quality_problems(quiz)
    report["final_problems"] = final_problems
    report["needs_polish"] = bool(final_problems)
    all_first = all(s["tries"] == 1 and s["ok"] for s in report["stages"])
    report["first_pass"] = all_first and not final_problems
    report["distribution"] = model.rarity(quiz)
    report["seconds"] = round(time.perf_counter() - started, 1)
    progress("done", f"完成（{report['seconds']}s）"
             + ("，一次通过" if report["first_pass"] else
                ("，仍有问题需人工润色：" + "；".join(final_problems[:3]) if final_problems else "，重试后通过")))
    return quiz, report


# ---------------------------------------------------------------- 渲染与索引

def build_gallery() -> list[dict[str, Any]]:
    """把预设 + 已生成 JSON 全部渲染到 docs（GitHub Pages 目录），并重建索引。"""
    quizzes: list[dict[str, Any]] = []
    for path in presets.preset_files():
        quizzes.append({"path": path, "quiz": model.load(path), "source": "preset"})
    for path in sorted(QUIZ_DIR.glob("*.json")):
        if path.stem.endswith(".report"):
            continue
        try:
            quizzes.append({"path": path, "quiz": model.load(path), "source": "generated"})
        except Exception as exc:  # noqa: BLE001
            _say(f"  跳过损坏的 {path.name}：{exc}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for item in quizzes:
        # 预设也要过校准：否则阈值走缺省值，分布会 skew 到 4.6%-26.4%
        quiz = model.calibrate(item["quiz"])
        render.write_quiz(quiz, DIST_DIR)
        manifest.append({
            "slug": quiz["slug"],
            "title": quiz["title"],
            "subtitle": quiz["subtitle"],
            "emoji": quiz["emoji"],
            "questions": len(quiz["questions"]),
            "source": item["source"],
        })
        _say(f"  ✓ {quiz['slug']}.html  {quiz['title']}")
    render.write_gallery(
        [{**item["quiz"], "source": item["source"]} for item in quizzes], DIST_DIR)
    MANIFEST.write_text(json.dumps({
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "count": len(manifest),
        "quizzes": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _say(f"✓ 索引已更新：{DIST_DIR / 'index.html'}（{len(manifest)} 个测试）")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Quiz Factory：一句话生成可分享的测试")
    parser.add_argument("--theme", help="测试主题，如「测测你是哪种打工人」")
    parser.add_argument("--slug", help="输出文件名（英文短横线），默认从主题生成")
    parser.add_argument("--preset", help="渲染内置预设（cat 等）")
    parser.add_argument("--all-presets", action="store_true", help="渲染全部预设并重建 gallery")
    parser.add_argument("--gallery", action="store_true", help="只重建 gallery")
    parser.add_argument("--list", action="store_true", help="列出内置预设")
    parser.add_argument("--count", type=int, default=8, help="题目数量（6-10，默认 8）")
    parser.add_argument("--provider", default="ollama", help="ollama（本地）或 openai（云端兼容）")
    parser.add_argument("--model", default="qwen2.5:7b", help="模型名")
    parser.add_argument("--base-url", default="", help="服务地址（默认本地 11434 / DeepSeek）")
    parser.add_argument("--api-key", default="", help="云端 API key（也可放 .env 的 QUIZ_API_KEY）")
    args = parser.parse_args()

    if args.list:
        for path in presets.preset_files():
            quiz = model.load(path)
            _say(f"  {path.stem:12s} {quiz['title']}（{len(quiz['questions'])} 题）")
        return 0
    if args.gallery or args.all_presets:
        build_gallery()
        return 0
    if args.preset:
        try:
            quiz = presets.load_preset(args.preset)
        except FileNotFoundError as exc:
            raise SystemExit(f"✗ {exc}") from exc
        path = render.write_quiz(quiz, DIST_DIR)
        build_gallery()
        _say(f"✓ 已渲染：{path}")
        return 0
    if not args.theme:
        parser.print_help()
        return 1

    slug = args.slug or model._slug(args.theme)
    try:
        quiz, report = generate(args.theme, slug, provider=args.provider, model_name=args.model,
                                base_url=args.base_url, api_key=args.api_key,
                                count=max(6, min(10, args.count)))
    except LLMError as exc:
        raise SystemExit(f"✗ 生成失败：{exc}") from exc
    saved = model.save(quiz, QUIZ_DIR / f"{slug}.json")
    (QUIZ_DIR / f"{slug}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _say(f"✓ 数据已保存：{saved}（门禁报告：{slug}.report.json）")
    build_gallery()
    return 0


if __name__ == "__main__":
    sys.exit(main())
