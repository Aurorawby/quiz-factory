"""检查渲染出来的 HTML：无占位符、结果/稀有度/深链都在、无外部资源。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import re  # noqa: E402

import model  # noqa: E402
import render  # noqa: E402

quiz = model.load(ROOT / "presets" / "cat.json")
html = render.render_quiz(quiz)
problems: list[str] = []

# 1. 占位符必须全部替换
leftover = re.findall(r"__[A-Z_]+__", html)
if leftover:
    problems.append(f"还有未替换的占位符：{set(leftover)}")

# 2. 8 种结果名都要在页面数据里
for key, result in quiz["results"].items():
    if result["name"] not in html:
        problems.append(f"结果 {key} 的名称没出现在 HTML 里")

# 3. 稀有度、深链、关键交互
for token in ("rarity", "#r=", "shareResult", "startQuiz", "resultKey"):
    if token not in html:
        problems.append(f"缺少关键逻辑：{token}")

# 4. 不能引用外部资源（单文件 / 离线可用）
external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
if external:
    problems.append(f"引用了外部资源：{external[:3]}")

# 5. 题目数量与结果数量对得上
if html.count('"questions"') != 1 or len(quiz["questions"]) != 8:
    problems.append("题目数据异常")

print(f"HTML 大小：{len(html) / 1024:.1f} KB")
if problems:
    print("✗ 发现问题：")
    for problem in problems:
        print("  -", problem)
    sys.exit(1)
print("✓ 渲染检查通过：无占位符 / 8 种结果齐全 / 稀有度与深链逻辑在位 / 无外部资源")
