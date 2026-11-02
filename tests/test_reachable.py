"""验证测试数据质量：8 个结果全部可达，且分布均衡（避免某种结果占 20%+）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json  # noqa: E402

import model  # noqa: E402

data = json.loads((ROOT / "presets" / "cat.json").read_text(encoding="utf-8"))
quiz = model.normalize(data, slug="cat")

problems = model.validate(quiz)
if problems:
    print("✗ 数据不合法：", problems)
    sys.exit(1)

quiz = model.calibrate(quiz)
print("自动校准阈值：", quiz.get("thresholds"))
shares = model.rarity(quiz)
print("结果分布（按均匀随机作答估算）：")
for key, share in sorted(shares.items()):
    print(f"  {key}  {quiz['results'][key]['name']:14s} {share:5.1f}%")

unreachable = [key for key, share in shares.items() if share <= 0]
too_common = [key for key, share in shares.items() if share > 18]
too_rare = [key for key, share in shares.items() if share < 7]
print(f"\n可达 {len(shares) - len(unreachable)}/8")
if unreachable:
    print("✗ 不可达:", unreachable)
if too_common:
    print("✗ 过于集中:", too_common)
if too_rare:
    print("✗ 过于罕见:", too_rare)
if unreachable or too_common or too_rare:
    sys.exit(1)
print("✓ 8 种结果全部可达，分布均衡（7%-18%）")
