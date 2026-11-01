"""Claude 代理客户端（可复用）

安全原则：
    - token 只在运行时从 ~/.claude/settings.json 读取，**绝不硬编码、不打印、不写入任何文件**。
    - 所有对外函数都不返回 token。

端点：Anthropic /v1/messages 兼容（本项目代理 = code1.newcli.com）。
默认模型：claude-sonnet-5（经 /v1/models 确认可用）。

用法：
    from _claude_client import call_claude, extract_json
    r = call_claude([{"role": "user", "content": "你好"}], system="你是助手")
    print(r["text"], r["usage"])
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
_DEFAULT_MODEL = "claude-sonnet-5"
_ANTHROPIC_VERSION = "2023-06-01"


def load_config() -> Dict[str, str]:
    """读取 Claude 代理配置（base_url / token / model）。token 仅内存使用。"""
    if not os.path.exists(_SETTINGS_PATH):
        raise RuntimeError(f"未找到 {_SETTINGS_PATH}")
    with open(_SETTINGS_PATH, encoding="utf-8") as f:
        env = (json.load(f) or {}).get("env", {})
    base = (env.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
    token = env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")
    model = env.get("ANTHROPIC_MODEL") or _DEFAULT_MODEL
    if not base or not token:
        raise RuntimeError("settings.json 缺少 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN")
    return {"base_url": base, "token": token, "model": model}


def call_claude(
    messages: List[Dict[str, Any]],
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    max_retries: int = 4,
    timeout: int = 180,
) -> Dict[str, Any]:
    """调用 Claude，返回 {"text": str, "usage": dict, "model": str}。

    对 429/5xx 做指数退避重试；其余 HTTPError 直接抛出（含代理返回的错误信息）。
    """
    cfg = load_config()
    body: Dict[str, Any] = {
        "model": model or cfg["model"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(cfg["base_url"] + "/v1/messages", data=data, method="POST")
            for k, v in {
                "content-type": "application/json",
                "anthropic-version": _ANTHROPIC_VERSION,
                "authorization": "Bearer " + cfg["token"],
                "x-api-key": cfg["token"],
            }.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = "".join(
                b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"
            )
            return {"text": text, "usage": payload.get("usage", {}), "model": payload.get("model")}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:200]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (408, 429, 500, 502, 503, 504, 529):
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Claude 调用失败（不可重试）: {last_err}")
        except Exception as e:  # 网络抖动等
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Claude 调用失败（已重试{max_retries}次）: {last_err}")


def extract_json(text: str) -> Any:
    """从模型输出中稳健提取 JSON（容忍 ```json 围栏与前后缀说明文字）。

    用 JSONDecoder.raw_decode 从第一个 { 或 [ 起解析首个合法 JSON 值，忽略尾部多余内容。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        # 去掉代码围栏行
        lines = [ln for ln in t.split("\n") if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从输出提取 JSON（前200字）: {text[:200]!r}")


if __name__ == "__main__":
    # 自检：确认端点连通 + JSON 提取
    r = call_claude(
        [{"role": "user", "content": '只输出一个JSON对象：{"ok": true, "n": 42}'}],
        max_tokens=100,
        temperature=0,
    )
    print("连通OK | model:", r["model"], "| usage:", r.get("usage", {}).get("output_tokens"), "out-tokens")
    print("提取JSON:", extract_json(r["text"]))
