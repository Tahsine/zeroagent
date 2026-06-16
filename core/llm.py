"""
core/llm.py — LLM Client HTTP pur.

Supporte tout provider OpenAI-compatible + Anthropic natif.
Zéro dépendance externe : http.client + json + urllib.parse uniquement.

Providers testés :
  - OpenAI             (base_url="https://api.openai.com/v1")
  - Anthropic          (base_url="https://api.anthropic.com/v1")
  - Kaggle LLM Server  (base_url="https://ton-tunnel.trycloudflare.com/v1")
  - Ollama             (base_url="http://localhost:11434/v1")
  - tout provider OpenAI-compatible
"""

import http.client
import json
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Un message dans une conversation."""
    role: str        # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None   # rempli si role == "tool"
    name: str | None = None           # nom du tool appelé

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolCall:
    """Une décision du LLM d'appeler un outil."""
    id: str
    name: str
    arguments: dict   # déjà parsé depuis JSON


@dataclass
class LLMResponse:
    """Réponse normalisée, indépendante du provider."""
    content: str                          # texte généré
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning: str | None = None
    # ^ Raisonnement interne exposé par certains providers :
    #   - Anthropic Extended Thinking : bloc type="thinking"
    #   - Deepseek R1 / compatibles  : champ "reasoning_content"
    #   - OpenAI o1/o3/o4            : non exposé dans l'API → toujours None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None and len(self.reasoning) > 0


# ---------------------------------------------------------------------------
# Détection du provider
# ---------------------------------------------------------------------------

def _detect_provider(base_url: str) -> str:
    """
    Détermine le provider à partir de l'URL.
    Retourne "anthropic" ou "openai" (utilisé pour tout provider compatible).
    """
    if "anthropic.com" in base_url:
        return "anthropic"
    return "openai"


# ---------------------------------------------------------------------------
# Construction des requêtes
# ---------------------------------------------------------------------------

def _build_openai_request(
    messages: list[Message],
    model: str,
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [m.to_dict() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def _build_anthropic_request(
    messages: list[Message],
    model: str,
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> tuple[dict, str | None]:
    """
    Retourne (body, system_prompt).
    Anthropic sépare le system message du reste des messages.
    """
    system_prompt: str | None = None
    filtered: list[dict] = []

    for m in messages:
        if m.role == "system":
            system_prompt = m.content
        else:
            filtered.append(m.to_dict())

    body: dict = {
        "model": model,
        "messages": filtered,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if system_prompt:
        body["system"] = system_prompt
    if tools:
        # Anthropic utilise un format légèrement différent pour les tools
        body["tools"] = [_openai_tool_to_anthropic(t) for t in tools]

    return body, system_prompt


def _openai_tool_to_anthropic(tool: dict) -> dict:
    """Convertit un tool au format OpenAI vers le format Anthropic."""
    fn = tool.get("function", tool)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


# ---------------------------------------------------------------------------
# Parsing des réponses
# ---------------------------------------------------------------------------

def _parse_openai_response(data: dict) -> LLMResponse:
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""

    # Deepseek R1 et providers compatibles exposent le raisonnement ici
    reasoning = msg.get("reasoning_content") or None

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        tool_calls.append(ToolCall(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=args,
        ))

    usage = data.get("usage", {})
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        model=data.get("model", ""),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        reasoning=reasoning,
    )


def _parse_anthropic_response(data: dict) -> LLMResponse:
    content_blocks = data.get("content", [])
    text_content = ""
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        btype = block.get("type")
        if btype == "thinking":
            # Extended Thinking — raisonnement interne exposé
            reasoning_parts.append(block.get("thinking", ""))
        elif btype == "text":
            text_content += block.get("text", "")
        elif btype == "tool_use":
            tool_calls.append(ToolCall(
                id=block["id"],
                name=block["name"],
                arguments=block.get("input", {}),
            ))

    usage = data.get("usage", {})
    return LLMResponse(
        content=text_content,
        tool_calls=tool_calls,
        model=data.get("model", ""),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
    )


# ---------------------------------------------------------------------------
# Parsing SSE streaming
# ---------------------------------------------------------------------------

def _iter_sse_lines(raw_bytes: bytes) -> Iterator[str]:
    """Parse les lignes SSE depuis les bytes reçus."""
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def _stream_openai(conn: http.client.HTTPSConnection | http.client.HTTPConnection) -> Iterator[str]:
    """Yield les chunks de texte depuis un stream OpenAI."""
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode()}")

    for chunk in response:
        for data_str in _iter_sse_lines(chunk):
            if data_str == "[DONE]":
                return
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


# ---------------------------------------------------------------------------
# Connexion HTTP
# ---------------------------------------------------------------------------

def _make_connection(
    host: str,
    port: int,
    use_ssl: bool,
    timeout: float,
) -> http.client.HTTPSConnection | http.client.HTTPConnection:
    if use_ssl:
        ctx = ssl.create_default_context()
        return http.client.HTTPSConnection(host, port, context=ctx, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _do_request(
    conn: http.client.HTTPSConnection | http.client.HTTPConnection,
    method: str,
    path: str,
    body: dict,
    headers: dict,
) -> dict:
    """Envoie la requête et retourne le JSON parsé."""
    payload = json.dumps(body).encode("utf-8")
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    raw = response.read()

    if response.status not in (200, 201):
        raise RuntimeError(
            f"HTTP {response.status} from {path}: {raw.decode('utf-8', errors='replace')}"
        )

    return json.loads(raw)


# ---------------------------------------------------------------------------
# LLMClient — classe principale
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Client LLM HTTP pur. Zero dépendance externe.

    Compatible avec tout provider OpenAI-compatible,
    et avec l'API Anthropic native.

    Usage:
        # OpenAI
        client = LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o",
        )

        # Anthropic
        client = LLMClient(
            base_url="https://api.anthropic.com/v1",
            api_key="sk-ant-...",
            model="claude-sonnet-4-6",
        )

        # Kaggle LLM Server / Ollama / local
        client = LLMClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:7b",
        )

        response = client.complete([
            Message(role="user", content="Quelle est la capitale du Bénin ?")
        ])
        print(response.content)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._provider = _detect_provider(base_url)
        parsed = urllib.parse.urlparse(self.base_url)
        self._scheme = parsed.scheme          # "http" ou "https"
        self._host = parsed.hostname or ""
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_path = parsed.path         # ex: "/v1"

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Envoie les messages au LLM et retourne une LLMResponse normalisée.

        Args:
            messages: historique de la conversation
            tools:    liste de JSON Schema tools (format OpenAI)

        Returns:
            LLMResponse avec content et éventuels tool_calls
        """
        for attempt in range(self.max_retries):
            try:
                return self._complete_once(messages, tools)
            except RuntimeError as e:
                msg = str(e)
                # Retry sur 429 (rate limit) et 5xx (erreurs serveur)
                if attempt < self.max_retries - 1 and (
                    "429" in msg or "500" in msg or "502" in msg or "503" in msg
                ):
                    wait = self.retry_delay * (2 ** attempt)  # backoff exponentiel
                    time.sleep(wait)
                    continue
                raise

        # Jamais atteint, mais satisfait mypy
        raise RuntimeError("Max retries exceeded")

    def stream(
        self,
        messages: list[Message],
    ) -> Iterator[str]:
        """
        Stream la réponse token par token.
        Uniquement pour les providers OpenAI-compatible (pas Anthropic pour l'instant).

        Usage:
            for chunk in client.stream(messages):
                print(chunk, end="", flush=True)
        """
        if self._provider == "anthropic":
            # Fallback : on complete et on yield d'un coup
            response = self.complete(messages)
            yield response.content
            return

        body = _build_openai_request(
            messages, self.model, None, self.temperature, self.max_tokens, stream=True
        )
        headers = self._headers()
        conn = _make_connection(
            self._host, self._port,
            use_ssl=(self._scheme == "https"),
            timeout=self.timeout,
        )
        try:
            payload = json.dumps(body).encode("utf-8")
            conn.request("POST", self._path("/chat/completions"), body=payload, headers=headers)
            yield from _stream_openai(conn)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _complete_once(
        self,
        messages: list[Message],
        tools: list[dict] | None,
    ) -> LLMResponse:
        conn = _make_connection(
            self._host, self._port,
            use_ssl=(self._scheme == "https"),
            timeout=self.timeout,
        )
        try:
            if self._provider == "anthropic":
                return self._complete_anthropic(conn, messages, tools)
            return self._complete_openai(conn, messages, tools)
        finally:
            conn.close()

    def _complete_openai(
        self,
        conn,
        messages: list[Message],
        tools: list[dict] | None,
    ) -> LLMResponse:
        body = _build_openai_request(
            messages, self.model, tools, self.temperature, self.max_tokens, stream=False
        )
        data = _do_request(conn, "POST", self._path("/chat/completions"), body, self._headers())
        return _parse_openai_response(data)

    def _complete_anthropic(
        self,
        conn,
        messages: list[Message],
        tools: list[dict] | None,
    ) -> LLMResponse:
        body, _ = _build_anthropic_request(
            messages, self.model, tools, self.temperature, self.max_tokens, stream=False
        )
        data = _do_request(conn, "POST", self._path("/messages"), body, self._headers())
        return _parse_anthropic_response(data)

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _path(self, endpoint: str) -> str:
        """Construit le path complet : base_path + endpoint."""
        return self._base_path + endpoint

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self._provider!r}, "
            f"model={self.model!r}, "
            f"host={self._host!r})"
        )