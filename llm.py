"""极简 LLM 客户端（只用 httpx）：本地 Ollama + OpenAI 兼容云端接口。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE = "http://localhost:11434"
DEFAULT_CLOUD_BASE = "https://api.deepseek.com/v1"


class LLMError(RuntimeError):
    """模型调用失败。"""


def _say(msg: str) -> None:
    """进度输出：stdout 不是 tty（后台/重定向）时也安全。"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)


class Ollama:
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = DEFAULT_BASE,
                 timeout: float = 600.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat_json(self, system: str, user: str, *, temperature: float = 0.6,
                  max_tokens: int = 3000, retries: int = 2) -> dict[str, Any]:
        """调模型并解析 JSON（失败重试，去掉 ```json 围栏）。"""
        last_error = ""
        for attempt in range(retries + 1):
            raw = self._chat(system, user, temperature=temperature + attempt * 0.1,
                             max_tokens=max_tokens)
            try:
                return _parse_json(raw)
            except ValueError as exc:
                last_error = f"{exc}；原始输出前 200 字：{raw[:200]}"
        raise LLMError(f"模型没有返回合法 JSON：{last_error}")

    def _chat(self, system: str, user: str, *, temperature: float,
              max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "format": "json",
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"连不上 Ollama（{self.base_url}）：{exc}") from exc
        content = (data.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LLMError("模型返回为空")
        _say(f"    模型生成完成（{time.perf_counter() - started:.1f}s，{len(content)} 字）")
        return content


class OpenAICompat:
    """DeepSeek / OpenAI / 通义 等 OpenAI 兼容接口（chat/completions）。"""

    def __init__(self, model: str, base_url: str = DEFAULT_CLOUD_BASE,
                 api_key: str = "", timeout: float = 180.0) -> None:
        if not api_key:
            raise LLMError("云端模型需要 API key（--api-key 或环境变量 QUIZ_API_KEY / LIT_API_KEY）")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat_json(self, system: str, user: str, *, temperature: float = 0.6,
                  max_tokens: int = 3000, retries: int = 2) -> dict[str, Any]:
        last_error = ""
        for attempt in range(retries + 1):
            raw = self._chat(system, user, temperature=temperature + attempt * 0.1,
                             max_tokens=max_tokens)
            try:
                return _parse_json(raw)
            except ValueError as exc:
                last_error = f"{exc}；原始输出前 200 字：{raw[:200]}"
        raise LLMError(f"模型没有返回合法 JSON：{last_error}")

    def _chat(self, system: str, user: str, *, temperature: float,
              max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"云端接口调用失败（{self.base_url}）：{exc}") from exc
        choices = data.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        if not content.strip():
            raise LLMError("模型返回为空")
        _say(f"    模型生成完成（{time.perf_counter() - started:.1f}s，{len(content)} 字）")
        return content


def make_client(provider: str, *, model: str, base_url: str = "",
                api_key: str = "") -> Ollama | OpenAICompat:
    """按 provider 构建客户端；provider 取 ollama / openai（云端兼容）。"""
    if provider == "ollama":
        return Ollama(model=model, base_url=base_url or DEFAULT_BASE)
    if provider in ("openai", "cloud", "deepseek"):
        return OpenAICompat(model=model, base_url=base_url or DEFAULT_CLOUD_BASE,
                            api_key=api_key)
    raise LLMError(f"未知模型来源：{provider}（可选 ollama / openai）")


def load_env_keys() -> dict[str, str]:
    """从项目根 .env / 本地 .env 里找 API key（ QUIZ_* 优先，LIT_* 兜底）。"""
    env: dict[str, str] = {}
    for candidate in (Path(__file__).resolve().parent.parent / ".env",
                      Path(__file__).resolve().parent / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    env.update({k: v for k, v in os.environ.items()
                if k in ("QUIZ_LLM_PROVIDER", "QUIZ_MODEL", "QUIZ_API_KEY",
                         "QUIZ_API_BASE", "LIT_API_KEY", "LIT_API_BASE")})
    return env


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("没有找到 JSON 对象")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("顶层不是对象")
    return data
