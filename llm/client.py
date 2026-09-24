import json
import time
import urllib.error
import urllib.request

from data.config import LLM_CONFIG
from llm.prompts import PROBLEM_DEFINITION

def _resolve_endpoint(model_name, api_key=None):
    """Choose the provider endpoint from the model name.

    No endpoint configuration is needed: users only set model_name and
    api_key in data/config.py.
    """
    name = (model_name or "").lower()
    if api_key and str(api_key).startswith("sk-ws-"):
        return "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
    if "kimi" in name:
        return "https://api.moonshot.cn/anthropic/v1/messages"
    if "glm" in name or "chatglm" in name:
        return "https://open.bigmodel.cn/api/anthropic/v1/messages"
    # Legacy default keeps compatibility with previously configured GLM models.
    return "https://open.bigmodel.cn/api/anthropic/v1/messages"


class LLMClient:
    """Stateless Anthropic-compatible client.

    Each request is self-contained: it sends the core problem rules plus the
    current operation system prompt as the system message, and only the current
    user prompt as the conversation. No previous request or response history is
    carried over between API calls.
    """

    def __init__(self, config=None):
        config = config or LLM_CONFIG
        self.model_name = config["model_name"]
        self.api_key = config["api_key"]
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 2000)
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 3)
        self.core_rules = PROBLEM_DEFINITION
        # This Anthropic-compatible endpoint is directly reachable in the test
        # environment. Bypass any inherited HTTP(S) proxy so an inactive local
        # proxy does not turn into a WinError 10061 connection failure.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.stats = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "failed_calls": 0,
            "fallback_calls": 0,
        }
        # Debug printing toggle: when True, print each outgoing prompt and
        # incoming LLM response to the console for verification.
        self.debug_print = bool(config.get("debug_print", False))

    def generate(self, system_prompt, user_prompt):
        system_text = f"{self.core_rules}\n\n{system_prompt}"
        messages = [{"role": "user", "content": user_prompt}]
        if self.debug_print:
            try:
                print("\n--- LLM REQUEST START ---")
                print("SYSTEM:\n", system_text)
                print("USER PROMPT:\n", user_prompt)
                print("MESSAGES (current request only, length %d):" % len(messages))
                for m in messages:
                    role = m.get("role")
                    content = m.get("content")
                    print(role + ":", (content if isinstance(content, str) else str(content)))
                print("--- LLM REQUEST END ---\n")
            except Exception:
                pass
        body = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if not (self.api_key and str(self.api_key).startswith("sk-ws-")):
            body["temperature"] = self.temperature
        if system_text:
            body["system"] = system_text
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        last_error = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                req = urllib.request.Request(
                    _resolve_endpoint(self.model_name, self.api_key), data=data, headers=headers, method="POST"
                )
                with self.opener.open(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - started
                text = self._extract_text(payload)
                usage = payload.get("usage") or {}
                in_tokens = usage.get("input_tokens", 0)
                out_tokens = usage.get("output_tokens", 0)
                self.stats["calls"] += 1
                self.stats["input_tokens"] += in_tokens
                self.stats["output_tokens"] += out_tokens
                self.stats["total_tokens"] += in_tokens + out_tokens
                self.stats["latency_seconds"] += elapsed
                if self.debug_print:
                    try:
                        print("\n--- LLM RESPONSE START ---")
                        print(text)
                        print("--- LLM RESPONSE END ---\n")
                    except Exception:
                        pass
                return text
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                self.stats["failed_calls"] += 1
                if attempt < self.max_retries - 1:
                    time.sleep(0.4 * (attempt + 1))
        raise RuntimeError("LLM request failed after retries: %s" % (last_error,))

    def _extract_text(self, payload):
        content = payload.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts)
        return payload.get("text", "")

    def core_rules_includes(self, system_text: str) -> bool:
        """Return True if `system_text` contains the full core rules block.

        This helper isolates the substring check and makes tests easier.
        """
        try:
            return isinstance(system_text, str) and (self.core_rules.strip() in system_text)
        except Exception:
            return False
