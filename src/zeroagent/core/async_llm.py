"""
core/async_llm.py — LLM Client async pur.

Même surface que LLMClient mais toutes les méthodes sont des coroutines.
Zéro dépendance externe : asyncio + ssl + json + urllib.parse uniquement.

Fonctionnement :
  - asyncio.open_connection() pour les connexions TLS/TCP
  - Protocole HTTP/1.1 écrit à la main (headers + body)
  - SSE streaming via lecture ligne par ligne async

Pourquoi ne pas réutiliser LLMClient ?
  - http.client est entièrement bloquant (pas de hook asyncio)
  - Un mixin async ne suffit pas — la couche I/O doit être remplacée
  - On réutilise toutes les fonctions de parsing de core/llm.py
    (elles sont pures, sans I/O)
"""

from __future__ import annotations

import asyncio
import json
import ssl
import urllib.parse
from typing import AsyncIterator

from zeroagent.core.llm import (
    LLMResponse,
    Message,
    ToolCall,
    _build_anthropic_request,
    _build_openai_request,
    _detect_provider,
    _parse_anthropic_response,
    _parse_openai_response,
)


# ---------------------------------------------------------------------------
# Helpers HTTP/1.1 async
# ---------------------------------------------------------------------------

async def _open_connection(
    host: str,
    port: int,
    use_ssl: bool,
    timeout: float,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Ouvre une connexion TCP ou TLS async."""
    ctx = ssl.create_default_context() if use_ssl else None
    return await asyncio.wait_for(
        asyncio.open_connection(host, port, ssl=ctx),
        timeout=timeout,
    )


def _build_http_request(
    method: str,
    path: str,
    host: str,
    body: dict,
    headers: dict,
) -> bytes:
    """Construit une requête HTTP/1.1 complète en bytes."""
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    lines = [
        f"{method} {path} HTTP/1.1",
        f"Host: {host}",
        f"Content-Length: {len(payload)}",
        "Connection: close",
    ]
    for k, v in headers.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode("utf-8") + payload


async def _read_response(
    reader: asyncio.StreamReader,
    timeout: float,
) -> tuple[int, bytes]:
    """
    Lit la réponse HTTP complète.
    Retourne (status_code, body_bytes).
    Gère Transfer-Encoding: chunked et Content-Length.
    """
    # Lire les headers
    header_lines: list[str] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not decoded:
            break
        header_lines.append(decoded)

    if not header_lines:
        raise RuntimeError("Réponse HTTP vide")

    # Parser le status
    status_line = header_lines[0]
    parts = status_line.split(" ", 2)
    if len(parts) < 2:
        raise RuntimeError(f"Status line invalide: {status_line!r}")
    status_code = int(parts[1])

    # Parser les headers
    response_headers: dict[str, str] = {}
    for line in header_lines[1:]:
        if ":" in line:
            k, _, v = line.partition(":")
            response_headers[k.strip().lower()] = v.strip()

    # Lire le body
    if response_headers.get("transfer-encoding", "").lower() == "chunked":
        body = await _read_chunked(reader, timeout)
    elif "content-length" in response_headers:
        length = int(response_headers["content-length"])
        body = await asyncio.wait_for(reader.readexactly(length), timeout=timeout)
    else:
        body = await asyncio.wait_for(reader.read(), timeout=timeout)

    return status_code, body


async def _read_chunked(reader: asyncio.StreamReader, timeout: float) -> bytes:
    """Lit un body encodé en chunked transfer encoding."""
    chunks: list[bytes] = []
    while True:
        size_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        size = int(size_line.strip(), 16)
        if size == 0:
            break
        chunk = await asyncio.wait_for(reader.readexactly(size), timeout=timeout)
        chunks.append(chunk)
        await asyncio.wait_for(reader.readline(), timeout=timeout)  # CRLF
    return b"".join(chunks)


async def _do_async_request(
    host: str,
    port: int,
    use_ssl: bool,
    method: str,
    path: str,
    body: dict,
    headers: dict,
    timeout: float,
) -> dict:
    """Envoie une requête HTTP async et retourne le JSON parsé."""
    reader, writer = await _open_connection(host, port, use_ssl, timeout)
    try:
        request_bytes = _build_http_request(method, path, host, body, headers)
        writer.write(request_bytes)
        await asyncio.wait_for(writer.drain(), timeout=timeout)

        status, raw = await _read_response(reader, timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    if status not in (200, 201):
        raise RuntimeError(
            f"HTTP {status} from {path}: {raw.decode('utf-8', errors='replace')}"
        )

    return json.loads(raw)


async def _stream_openai_async(
    host: str,
    port: int,
    use_ssl: bool,
    path: str,
    body: dict,
    headers: dict,
    timeout: float,
) -> AsyncIterator[str]:
    """Yield les chunks de texte depuis un stream OpenAI (SSE) async."""
    reader, writer = await _open_connection(host, port, use_ssl, timeout)
    try:
        request_bytes = _build_http_request("POST", path, host, body, headers)
        writer.write(request_bytes)
        await asyncio.wait_for(writer.drain(), timeout=timeout)

        # Lire les headers HTTP
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if line in (b"\r\n", b"\n", b""):
                break

        # Lire les lignes SSE
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded.startswith("data:"):
                continue
            data_str = decoded[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AsyncLLMClient
# ---------------------------------------------------------------------------

class AsyncLLMClient:
    """
    Client LLM async. Zéro dépendance externe.

    Même interface que LLMClient mais toutes les méthodes sont des coroutines.
    Compatible avec tout provider OpenAI-compatible et Anthropic natif.

    Usage:
        client = AsyncLLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o",
        )

        response = await client.acomplete([
            Message(role="user", content="Quelle est la capitale du Bénin ?")
        ])
        print(response.content)

        # Streaming
        async for chunk in client.astream(messages):
            print(chunk, end="", flush=True)

    Note : AsyncLLMClient peut être construit depuis un LLMClient existant
    via AsyncLLMClient.from_sync(client).
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
    ) -> None:
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
        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_path = parsed.path

    # ------------------------------------------------------------------
    # Constructeur alternatif depuis LLMClient
    # ------------------------------------------------------------------

    @classmethod
    def from_sync(cls, client) -> "AsyncLLMClient":
        """
        Crée un AsyncLLMClient depuis un LLMClient existant.
        Pratique pour migrer progressivement sans changer la config.

        Usage:
            sync_client = LLMClient(base_url="...", model="...", api_key="...")
            async_client = AsyncLLMClient.from_sync(sync_client)
        """
        return cls(
            base_url=client.base_url,
            model=client.model,
            api_key=client.api_key,
            temperature=client.temperature,
            max_tokens=client.max_tokens,
            timeout=client.timeout,
            max_retries=client.max_retries,
            retry_delay=client.retry_delay,
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Envoie les messages au LLM et retourne une LLMResponse normalisée.
        Retry automatique sur 429/5xx avec backoff exponentiel.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self._acomplete_once(messages, tools)
            except RuntimeError as e:
                last_error = e
                msg = str(e)
                if attempt < self.max_retries - 1 and (
                    "429" in msg or "500" in msg or "502" in msg or "503" in msg
                ):
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise
        raise last_error or RuntimeError("Max retries exceeded")

    async def astream(
        self,
        messages: list[Message],
    ) -> AsyncIterator[str]:
        """
        Stream la réponse token par token (async generator).

        Usage:
            async for chunk in client.astream(messages):
                print(chunk, end="", flush=True)
        """
        if self._provider == "anthropic":
            # Fallback : complete et yield en une fois
            response = await self.acomplete(messages)
            yield response.content
            return

        body = _build_openai_request(
            messages, self.model, None,
            self.temperature, self.max_tokens, stream=True,
        )
        async for chunk in _stream_openai_async(
            self._host, self._port,
            use_ssl=(self._scheme == "https"),
            path=self._path("/chat/completions"),
            body=body,
            headers=self._headers(),
            timeout=self.timeout,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    async def _acomplete_once(
        self,
        messages: list[Message],
        tools: list[dict] | None,
    ) -> LLMResponse:
        if self._provider == "anthropic":
            return await self._acomplete_anthropic(messages, tools)
        return await self._acomplete_openai(messages, tools)

    async def _acomplete_openai(
        self,
        messages: list[Message],
        tools: list[dict] | None,
    ) -> LLMResponse:
        body = _build_openai_request(
            messages, self.model, tools,
            self.temperature, self.max_tokens, stream=False,
        )
        data = await _do_async_request(
            self._host, self._port,
            use_ssl=(self._scheme == "https"),
            method="POST",
            path=self._path("/chat/completions"),
            body=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return _parse_openai_response(data)

    async def _acomplete_anthropic(
        self,
        messages: list[Message],
        tools: list[dict] | None,
    ) -> LLMResponse:
        body, _ = _build_anthropic_request(
            messages, self.model, tools,
            self.temperature, self.max_tokens, stream=False,
        )
        data = await _do_async_request(
            self._host, self._port,
            use_ssl=(self._scheme == "https"),
            method="POST",
            path=self._path("/messages"),
            body=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
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
        return self._base_path + endpoint

    def __repr__(self) -> str:
        return (
            f"AsyncLLMClient(provider={self._provider!r}, "
            f"model={self.model!r}, "
            f"host={self._host!r})"
        )
