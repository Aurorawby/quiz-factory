"""对所有测试数据做统一体检：结构合法 + 8 种结果可达 + 校准后分布均衡。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model  # noqa: E402
import presets  # noqa: E402

failures: list[str] = []
generated = [p for p in (ROOT / "quizzes").glob("*.json") if not p.stem.endswith(".report")]
for path in sorted(presets.preset_files()) + sorted(generated):
    try:
        quiz = model.load(path)
        quiz = model.calibrate(quiz)
        problems = model.validate(quiz) + model.quality_problems(quiz)
        shares = model.rarity(quiz)
        name = path.stem
        if problems:
            failures.append(f"{name}: {'；'.join(problems[:3])}")
        if not shares or min(shares.values()) <= 0:
            failures.append(f"{name}: 有结果不可达")
        print(f"  {name:8s} 题目 {len(quiz['questions'])}  "
              f"分布 {min(shares.values()):.1f}%-{max(shares.values()):.1f}%  "
              f"阈值 {quiz.get('thresholds')}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

print()
if failures:
    print("✗ 未通过：")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("✓ 全部测试数据通过：结构合法、8 种结果可达、校准后分布均衡")
