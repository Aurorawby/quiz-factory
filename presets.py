"""预设测试：保证没有模型也能演示。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model import load

PRESET_DIR = Path(__file__).resolve().parent / "presets"


def preset_files() -> list[Path]:
    return sorted(PRESET_DIR.glob("*.json"))


def load_preset(slug: str) -> dict[str, Any]:
    path = PRESET_DIR / f"{slug}.json"
    if not path.exists():
        names = ", ".join(p.stem for p in preset_files())
        raise FileNotFoundError(f"没有名为 {slug!r} 的预设，可用：{names}")
    return load(path)


def load_all() -> list[dict[str, Any]]:
    return [load(path) for path in preset_files()]
