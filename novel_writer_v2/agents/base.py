"""Base agent with API routing for Gemini and Qwen."""

import time
from dataclasses import dataclass, field

from ..config import GeminiConfig, QwenConfig
from ..utils import parse_json_response


@dataclass
class AgentLog:
    agent_name: str = ""
    action: str = ""
    prompt_preview: str = ""
    response_preview: str = ""
    elapsed_seconds: float = 0.0


class BaseAgent:
    """Base class for all agents with Gemini and Qwen API access."""

    def __init__(
        self,
        name: str,
        gemini_config: GeminiConfig,
        qwen_config: QwenConfig | None = None,
    ):
        self.name = name
        self.gemini_config = gemini_config
        self.qwen_config = qwen_config
        self.logs: list[AgentLog] = []

    def call_gemini(
        self,
        system: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call Gemini API via google-genai SDK."""
        from google import genai
        from google.genai import types

        start = time.time()
        client = genai.Client(api_key=self.gemini_config.api_key)
        response = client.models.generate_content(
            model=self.gemini_config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature or self.gemini_config.temperature,
                max_output_tokens=max_tokens or self.gemini_config.max_output_tokens,
            ),
        )
        result = response.text
        self._log("call_gemini", prompt, result, time.time() - start)
        return result

    def call_qwen(
        self,
        system: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call Qwen API via OpenAI-compatible DashScope endpoint."""
        if not self.qwen_config:
            raise ValueError(f"Agent {self.name} has no Qwen config")

        from openai import OpenAI

        start = time.time()
        client = OpenAI(
            api_key=self.qwen_config.api_key,
            base_url=self.qwen_config.base_url,
        )
        response = client.chat.completions.create(
            model=self.qwen_config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature or self.qwen_config.temperature,
            max_tokens=max_tokens or self.qwen_config.max_tokens,
        )
        result = response.choices[0].message.content
        self._log("call_qwen", prompt, result, time.time() - start)
        return result

    def call_gemini_json(
        self,
        system: str,
        prompt: str,
        temperature: float | None = None,
    ) -> dict:
        """Call Gemini and parse JSON from the response."""
        raw = self.call_gemini(
            system + "\n\nRespond with valid JSON only.",
            prompt,
            temperature=temperature,
        )
        return parse_json_response(raw)

    def _log(
        self, action: str, prompt: str, response: str, elapsed: float
    ) -> None:
        self.logs.append(
            AgentLog(
                agent_name=self.name,
                action=action,
                prompt_preview=prompt[:200],
                response_preview=response[:200] if response else "",
                elapsed_seconds=round(elapsed, 2),
            )
        )
