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
    """Minimal Anthropic-compatible client with persistent task memory.

    Core behavioral rules are sent only on the first call. Subsequent calls keep
    the prior conversation context and only add the current operation-specific
    request data. If the accumulated context grows too large, the oldest history
    entries are trimmed and the core rules are reinserted as a short reminder.
    """

    def __init__(self, config=None):
        config = config or LLM_CONFIG
        self.model_name = config["model_name"]
        self.api_key = config["api_key"]
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 2000)
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 3)
        self.max_history_chars = 12000
        self.history = []
        self.core_rules = PROBLEM_DEFINITION
        self.core_reminder = (
            "CORE RULES REMINDER: keep the original task-order and feasibility rules. "
            "Do not change task order, task attributes, or OT set. Keep previous "
            "context while responding to the current request."
        )
        # The first system prompt sets the task-specific operating mode (selection /
        # crossover / mutation). After initialization, do not keep re-injecting a
        # different operation prompt on each batch call. Only the core rules + the
        # initial operation reminder may be re-sent if the conversation becomes too
        # long, matching the same "one-time + reminder" semantics as PROBLEM_DEFINITION.
        self._initialized = False
        self._initial_system_prompt = None
        # Track whether the full core rules text has been printed to console
        # to ensure we only print the long PROBLEM_DEFINITION once.
        self._core_rules_printed = False
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

    def _history_text(self):
        text = ""
        for msg in self.history:
            if isinstance(msg.get("content"), str):
                text += msg["content"]
            else:
                for item in msg.get("content", []):
                    if isinstance(item, str):
                        text += item
                    elif isinstance(item, dict):
                        text += str(item.get("text", ""))
        return text

    def _trim_history_if_needed(self):
        if len(self._history_text()) <= self.max_history_chars:
            return
        while len(self._history_text()) > self.max_history_chars and len(self.history) > 2:
            self.history.pop(0)
        if len(self._history_text()) > self.max_history_chars:
            self.history = [
                {"role": "user", "content": self.core_reminder},
                *self.history[-max(2, len(self.history) - 1):],
            ]

    def _build_request(self, system_prompt, user_prompt):
        current_user = {"role": "user", "content": user_prompt}
        if not self._initialized:
            self._initialized = True
            self.history = []
            self._initial_system_prompt = system_prompt
            system_text = f"{self.core_rules}\n\n{self._initial_system_prompt}"
            messages = [current_user]
            return system_text, messages

        self._trim_history_if_needed()
        messages = list(self.history) + [current_user]
        if len(self._history_text()) > self.max_history_chars * 0.8:
            reminder_prompt = self._initial_system_prompt or system_prompt
            system_text = f"{self.core_rules}\n\nReminder: {reminder_prompt}"
        else:
            system_text = None
        return system_text, messages

    def generate(self, system_prompt, user_prompt):
        system_text, messages = self._build_request(system_prompt, user_prompt)
        if self.debug_print:
            try:
                print("\n--- LLM REQUEST START ---")
                # Print system text only when it's actually included in the
                # outgoing request. If it contains the full core rules text,
                # print it only the first time it is sent.
                if system_text:
                    if isinstance(system_text, str) and self.core_rules_includes(system_text):
                        if not self._core_rules_printed:
                            print("SYSTEM:\n", system_text)
                            self._core_rules_printed = True
                        else:
                            # Core rules would be resent (e.g. as a reminder);
                            # suppress re-printing the large block and show a
                            # brief note instead.
                            print("SYSTEM: (core rules suppressed; reminder sent)")
                    else:
                        print("SYSTEM:\n", system_text)
                print("USER PROMPT:\n", user_prompt)
                print("MESSAGES (history length %d):" % len(messages))
                for m in messages[-8:]:
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
                self.history.append({"role": "user", "content": user_prompt})
                self.history.append({"role": "assistant", "content": text})
                self._trim_history_if_needed()
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
