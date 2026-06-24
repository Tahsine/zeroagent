"""
tests/test_llm.py

Tests unitaires pour core/llm.py.
Zero dépendance : unittest + unittest.mock uniquement.

Stratégie : on mock http.client.HTTPSConnection et HTTPConnection
pour simuler les réponses réseau sans faire de vrais appels API.
"""

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from zeroagent.core.llm import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    _build_anthropic_request,
    _build_openai_request,
    _detect_provider,
    _messages_to_anthropic,
    _openai_tool_to_anthropic,
    _parse_anthropic_response,
    _parse_openai_response,
)


# ---------------------------------------------------------------------------
# Helpers — fabriquer des fausses réponses HTTP
# ---------------------------------------------------------------------------

def _make_http_response(status: int, body: dict) -> MagicMock:
    """Simule un objet http.client.HTTPResponse."""
    mock = MagicMock()
    mock.status = status
    mock.read.return_value = json.dumps(body).encode("utf-8")
    return mock


OPENAI_RESPONSE = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Porto-Novo est la capitale du Bénin.",
            "tool_calls": None,
        },
        "finish_reason": "stop",
    }],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 10,
    },
}

OPENAI_TOOL_RESPONSE = {
    "id": "chatcmpl-xyz",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "capitale Bénin"}',
                },
            }],
        },
        "finish_reason": "tool_calls",
    }],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
}

ANTHROPIC_RESPONSE = {
    "id": "msg_abc",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {"type": "text", "text": "Porto-Novo est la capitale officielle du Bénin."}
    ],
    "usage": {"input_tokens": 12, "output_tokens": 11},
}


ANTHROPIC_THINKING_RESPONSE = {
    "id": "msg_think",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "thinking",
            "thinking": "Porto-Novo est la capitale officielle, mais Cotonou est le siège économique...",
        },
        {"type": "text", "text": "Porto-Novo est la capitale du Bénin."},
    ],
    "usage": {"input_tokens": 20, "output_tokens": 15},
}

OPENAI_REASONING_RESPONSE = {
    "id": "chatcmpl-r1",
    "object": "chat.completion",
    "model": "deepseek-reasoner",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Porto-Novo est la capitale du Bénin.",
            "reasoning_content": "Je dois identifier la capitale du Bénin. C'est Porto-Novo.",
            "tool_calls": None,
        },
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
}

ANTHROPIC_TOOL_RESPONSE = {
    "id": "msg_xyz",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_001",
            "name": "search",
            "input": {"query": "capitale Bénin"},
        }
    ],
    "usage": {"input_tokens": 18, "output_tokens": 9},
}


# ---------------------------------------------------------------------------
# Tests : détection du provider
# ---------------------------------------------------------------------------

class TestDetectProvider(unittest.TestCase):

    def test_anthropic_url(self):
        self.assertEqual(_detect_provider("https://api.anthropic.com/v1"), "anthropic")

    def test_openai_url(self):
        self.assertEqual(_detect_provider("https://api.openai.com/v1"), "openai")

    def test_local_ollama(self):
        self.assertEqual(_detect_provider("http://localhost:11434/v1"), "openai")

    def test_cloudflare_tunnel(self):
        self.assertEqual(_detect_provider("https://mon-tunnel.trycloudflare.com/v1"), "openai")


# ---------------------------------------------------------------------------
# Tests : construction des requêtes
# ---------------------------------------------------------------------------

class TestBuildRequests(unittest.TestCase):

    def setUp(self):
        self.messages = [
            Message(role="system", content="Tu es un assistant utile."),
            Message(role="user", content="Quelle est la capitale du Bénin ?"),
        ]

    def test_openai_request_basic(self):
        body = _build_openai_request(
            self.messages, "gpt-4o", None, 0.7, 1024, False
        )
        self.assertEqual(body["model"], "gpt-4o")
        self.assertEqual(len(body["messages"]), 2)
        self.assertFalse(body["stream"])
        self.assertNotIn("tools", body)

    def test_openai_request_with_tools(self):
        tools = [{"function": {"name": "search", "description": "Recherche", "parameters": {}}}]
        body = _build_openai_request(
            self.messages, "gpt-4o", tools, 0.7, 1024, False
        )
        self.assertIn("tools", body)
        self.assertEqual(body["tool_choice"], "auto")

    def test_anthropic_request_extracts_system(self):
        body, system = _build_anthropic_request(
            self.messages, "claude-sonnet-4-6", None, 0.7, 1024, False
        )
        # Le message system doit être extrait du tableau messages
        self.assertEqual(system, "Tu es un assistant utile.")
        self.assertEqual(body["system"], "Tu es un assistant utile.")
        # Il ne doit plus être dans messages[]
        roles = [m["role"] for m in body["messages"]]
        self.assertNotIn("system", roles)
        self.assertEqual(len(body["messages"]), 1)

    def test_anthropic_request_max_tokens_required(self):
        body, _ = _build_anthropic_request(
            self.messages, "claude-sonnet-4-6", None, 0.7, 512, False
        )
        self.assertEqual(body["max_tokens"], 512)

    def test_anthropic_assistant_tool_call_becomes_tool_use_block(self):
        """
        Régression : un message assistant avec tool_calls doit devenir
        un bloc content de type 'tool_use', pas un champ 'tool_calls'
        séparé (qui n'existe pas dans l'API Anthropic).
        """
        tc = ToolCall(id="toolu_01", name="search", arguments={"query": "Bénin"})
        messages = [
            Message(role="user", content="Cherche la capitale du Bénin"),
            Message(role="assistant", content="", tool_calls=[tc]),
        ]
        result = _messages_to_anthropic(messages)

        assistant_msg = result[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIsInstance(assistant_msg["content"], list)
        tool_use_block = assistant_msg["content"][0]
        self.assertEqual(tool_use_block["type"], "tool_use")
        self.assertEqual(tool_use_block["id"], "toolu_01")
        self.assertEqual(tool_use_block["name"], "search")
        self.assertEqual(tool_use_block["input"], {"query": "Bénin"})

    def test_anthropic_tool_result_becomes_user_message(self):
        """
        Régression : un message role='tool' doit devenir un message
        role='user' avec un bloc 'tool_result' — Anthropic n'a pas
        de role='tool'. Sans cette conversion, l'API Anthropic
        rejette la requête ou ignore l'observation.
        """
        messages = [
            Message(role="tool", content="Porto-Novo", tool_call_id="toolu_01", name="search"),
        ]
        result = _messages_to_anthropic(messages)

        self.assertEqual(result[0]["role"], "user")
        block = result[0]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "toolu_01")
        self.assertEqual(block["content"], "Porto-Novo")

    def test_anthropic_full_tool_cycle_in_request(self):
        """Le cycle complet assistant→tool_use puis tool→tool_result dans une requête."""
        tc = ToolCall(id="toolu_01", name="search", arguments={"query": "test"})
        messages = [
            Message(role="system", content="Tu es utile."),
            Message(role="user", content="Question"),
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content="Résultat", tool_call_id="toolu_01", name="search"),
        ]
        body, system = _build_anthropic_request(
            messages, "claude-sonnet-4-6", None, 0.7, 1024, False
        )
        self.assertEqual(system, "Tu es utile.")
        # system extrait, donc 3 messages restants : user, assistant, tool→user
        self.assertEqual(len(body["messages"]), 3)
        self.assertEqual(body["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(body["messages"][2]["content"][0]["type"], "tool_result")

    def test_openai_tool_to_anthropic_conversion(self):
        openai_tool = {
            "function": {
                "name": "search",
                "description": "Effectue une recherche web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "La requête"}
                    },
                    "required": ["query"],
                },
            }
        }
        result = _openai_tool_to_anthropic(openai_tool)
        self.assertEqual(result["name"], "search")
        self.assertEqual(result["description"], "Effectue une recherche web")
        self.assertIn("input_schema", result)


# ---------------------------------------------------------------------------
# Tests : parsing des réponses
# ---------------------------------------------------------------------------

class TestParseResponses(unittest.TestCase):

    def test_parse_openai_text_response(self):
        response = _parse_openai_response(OPENAI_RESPONSE)
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Porto-Novo est la capitale du Bénin.")
        self.assertEqual(response.model, "gpt-4o")
        self.assertEqual(response.input_tokens, 15)
        self.assertEqual(response.output_tokens, 10)
        self.assertFalse(response.has_tool_calls)

    def test_parse_openai_tool_call_response(self):
        response = _parse_openai_response(OPENAI_TOOL_RESPONSE)
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(len(response.tool_calls), 1)
        tc = response.tool_calls[0]
        self.assertIsInstance(tc, ToolCall)
        self.assertEqual(tc.id, "call_001")
        self.assertEqual(tc.name, "search")
        self.assertEqual(tc.arguments, {"query": "capitale Bénin"})

    def test_parse_anthropic_text_response(self):
        response = _parse_anthropic_response(ANTHROPIC_RESPONSE)
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Porto-Novo est la capitale officielle du Bénin.")
        self.assertEqual(response.model, "claude-sonnet-4-6")
        self.assertEqual(response.input_tokens, 12)
        self.assertEqual(response.output_tokens, 11)
        self.assertFalse(response.has_tool_calls)

    def test_parse_anthropic_tool_call_response(self):
        response = _parse_anthropic_response(ANTHROPIC_TOOL_RESPONSE)
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(len(response.tool_calls), 1)
        tc = response.tool_calls[0]
        self.assertEqual(tc.id, "toolu_001")
        self.assertEqual(tc.name, "search")
        self.assertEqual(tc.arguments, {"query": "capitale Bénin"})

    def test_parse_anthropic_thinking_response(self):
        """Extended Thinking : le bloc thinking doit remplir response.reasoning."""
        response = _parse_anthropic_response(ANTHROPIC_THINKING_RESPONSE)
        self.assertEqual(response.content, "Porto-Novo est la capitale du Bénin.")
        self.assertTrue(response.has_reasoning)
        self.assertIn("Porto-Novo", response.reasoning)
        self.assertIn("Cotonou", response.reasoning)

    def test_parse_openai_reasoning_content(self):
        """Deepseek R1 / providers compatibles : reasoning_content doit être extrait."""
        response = _parse_openai_response(OPENAI_REASONING_RESPONSE)
        self.assertEqual(response.content, "Porto-Novo est la capitale du Bénin.")
        self.assertTrue(response.has_reasoning)
        self.assertIn("Porto-Novo", response.reasoning)

    def test_no_reasoning_by_default(self):
        """Une réponse standard sans reasoning doit avoir has_reasoning == False."""
        response = _parse_openai_response(OPENAI_RESPONSE)
        self.assertFalse(response.has_reasoning)
        self.assertIsNone(response.reasoning)

    def test_parse_openai_malformed_tool_args(self):
        """Les arguments JSON malformés ne doivent pas crasher."""
        bad = json.loads(json.dumps(OPENAI_TOOL_RESPONSE))
        bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "NOT JSON {"
        response = _parse_openai_response(bad)
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0].arguments, {})


# ---------------------------------------------------------------------------
# Tests : LLMClient — mock réseau complet
# ---------------------------------------------------------------------------

class TestLLMClientOpenAI(unittest.TestCase):

    def _make_client(self) -> LLMClient:
        return LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

    @patch("http.client.HTTPSConnection")
    def test_complete_text(self, MockConn):
        mock_conn = MockConn.return_value.__enter__.return_value
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(200, OPENAI_RESPONSE)

        client = self._make_client()
        response = client.complete([Message(role="user", content="Capitale du Bénin ?")])

        self.assertEqual(response.content, "Porto-Novo est la capitale du Bénin.")
        self.assertFalse(response.has_tool_calls)

    @patch("http.client.HTTPSConnection")
    def test_complete_with_tool_calls(self, MockConn):
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(200, OPENAI_TOOL_RESPONSE)

        client = self._make_client()
        tools = [{"function": {"name": "search", "description": "Search", "parameters": {}}}]
        response = client.complete(
            [Message(role="user", content="Capitale du Bénin ?")],
            tools=tools,
        )

        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0].name, "search")

    @patch("http.client.HTTPSConnection")
    def test_retry_on_500(self, MockConn):
        """Doit retenter sur HTTP 500 et réussir au 2e essai."""
        mock_conn = MockConn.return_value
        mock_conn.getresponse.side_effect = [
            _make_http_response(500, {"error": "Internal Server Error"}),
            _make_http_response(200, OPENAI_RESPONSE),
        ]

        client = LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            max_retries=3,
            retry_delay=0,   # pas de délai en test
        )
        response = client.complete([Message(role="user", content="test")])
        self.assertEqual(response.content, "Porto-Novo est la capitale du Bénin.")
        self.assertEqual(mock_conn.getresponse.call_count, 2)

    @patch("http.client.HTTPSConnection")
    def test_raises_after_max_retries(self, MockConn):
        """Doit lever RuntimeError après épuisement des retries."""
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(
            500, {"error": "always failing"}
        )

        client = LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            max_retries=2,
            retry_delay=0,
        )
        with self.assertRaises(RuntimeError):
            client.complete([Message(role="user", content="test")])

    @patch("http.client.HTTPSConnection")
    def test_headers_openai(self, MockConn):
        """Vérifie que le header Authorization est bien envoyé."""
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(200, OPENAI_RESPONSE)

        client = self._make_client()
        client.complete([Message(role="user", content="test")])

        call_args = mock_conn.request.call_args
        headers = call_args[1]["headers"] if "headers" in call_args[1] else call_args[0][3]
        self.assertIn("Authorization", headers)
        self.assertTrue(headers["Authorization"].startswith("Bearer "))


class TestLLMClientAnthropic(unittest.TestCase):

    def _make_client(self) -> LLMClient:
        return LLMClient(
            base_url="https://api.anthropic.com/v1",
            api_key="sk-ant-test",
            model="claude-sonnet-4-6",
        )

    @patch("http.client.HTTPSConnection")
    def test_complete_text(self, MockConn):
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(200, ANTHROPIC_RESPONSE)

        client = self._make_client()
        response = client.complete([Message(role="user", content="Capitale du Bénin ?")])

        self.assertEqual(response.content, "Porto-Novo est la capitale officielle du Bénin.")
        self.assertFalse(response.has_tool_calls)

    @patch("http.client.HTTPSConnection")
    def test_complete_with_tool_calls(self, MockConn):
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(200, ANTHROPIC_TOOL_RESPONSE)

        client = self._make_client()
        response = client.complete(
            [Message(role="user", content="Capitale du Bénin ?")],
            tools=[{"function": {"name": "search", "description": "Search", "parameters": {}}}],
        )

        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0].name, "search")

    @patch("http.client.HTTPSConnection")
    def test_headers_anthropic(self, MockConn):
        """Vérifie que x-api-key et anthropic-version sont présents."""
        mock_conn = MockConn.return_value
        mock_conn.getresponse.return_value = _make_http_response(200, ANTHROPIC_RESPONSE)

        client = self._make_client()
        client.complete([Message(role="user", content="test")])

        call_args = mock_conn.request.call_args
        headers = call_args[1]["headers"] if "headers" in call_args[1] else call_args[0][3]
        self.assertIn("x-api-key", headers)
        self.assertIn("anthropic-version", headers)
        self.assertNotIn("Authorization", headers)


# ---------------------------------------------------------------------------
# Tests : Message dataclass
# ---------------------------------------------------------------------------

class TestMessage(unittest.TestCase):

    def test_to_dict_basic(self):
        m = Message(role="user", content="Bonjour")
        d = m.to_dict()
        self.assertEqual(d, {"role": "user", "content": "Bonjour"})

    def test_to_dict_tool_result(self):
        m = Message(role="tool", content="Paris", tool_call_id="call_001", name="search")
        d = m.to_dict()
        self.assertEqual(d["tool_call_id"], "call_001")
        self.assertEqual(d["name"], "search")

    def test_to_dict_no_none_fields(self):
        """Les champs None ne doivent pas apparaître dans le dict."""
        m = Message(role="user", content="test")
        d = m.to_dict()
        self.assertNotIn("tool_call_id", d)
        self.assertNotIn("name", d)


# ---------------------------------------------------------------------------
# Tests : LLMClient repr
# ---------------------------------------------------------------------------

class TestLLMClientRepr(unittest.TestCase):

    def test_repr(self):
        client = LLMClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        r = repr(client)
        self.assertIn("openai", r)
        self.assertIn("gpt-4o", r)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
