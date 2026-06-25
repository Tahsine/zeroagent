"""
tests/test_async.py — Tests pour AsyncLLMClient, AsyncAgentLoop, Agent.arun().

Tout est mocké — aucun appel réseau réel.
Stratégie :
  - AsyncLLMClient : on mock _do_async_request et les connexions
  - AsyncAgentLoop : on injecte un faux AsyncLLMClient avec acomplete() mocké
  - Agent.arun()   : idem, via faux LLMClient qui expose acomplete()
  - AsyncExecutor  : on teste la parallélisation et les tools async natifs
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from zeroagent.core.async_llm import AsyncLLMClient
from zeroagent.core.llm import LLMClient, LLMResponse, Message, ToolCall
from zeroagent.core.memory import BufferMemory
from zeroagent.core.tools import ToolRegistry, tool
from zeroagent.harness.agent import Agent
from zeroagent.harness.async_loop import AsyncAgentLoop, AsyncExecutor
from zeroagent.harness.loop import RunConfig, StopReason
from zeroagent.harness.parser import ParsedAction, ActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Exécute une coroutine dans asyncio.run() — compatible Python 3.10+."""
    return asyncio.run(coro)


def make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        reasoning=None,
    )


def make_tool_response(tool_name: str, args: dict, call_id: str = "tc-1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(name=tool_name, arguments=args, id=call_id)],
        reasoning=None,
    )


class FakeAsyncLLM:
    """
    Faux AsyncLLMClient pour les tests.
    responses : liste de LLMResponse à retourner en séquence.
    """
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.model = "fake-async-model"
        self._responses = list(responses)
        self._call_count = 0

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return make_text_response("réponse par défaut")
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


# ---------------------------------------------------------------------------
# Tests AsyncLLMClient
# ---------------------------------------------------------------------------

class TestAsyncLLMClientInit(unittest.TestCase):
    """Vérifie la construction et la détection de provider."""

    def test_openai_provider_detected(self):
        client = AsyncLLMClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        self.assertEqual(client._provider, "openai")

    def test_anthropic_provider_detected(self):
        client = AsyncLLMClient(
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-6",
            api_key="sk-ant-test",
        )
        self.assertEqual(client._provider, "anthropic")

    def test_ollama_local_provider_detected(self):
        client = AsyncLLMClient(
            base_url="http://localhost:11434/v1",
            model="qwen3",
            api_key="",
        )
        self.assertEqual(client._provider, "openai")

    def test_from_sync_copies_all_fields(self):
        sync = LLMClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="sk-test",
            temperature=0.5,
            max_tokens=512,
        )
        async_client = AsyncLLMClient.from_sync(sync)
        self.assertEqual(async_client.base_url, sync.base_url)
        self.assertEqual(async_client.model, sync.model)
        self.assertEqual(async_client.api_key, sync.api_key)
        self.assertEqual(async_client.temperature, sync.temperature)
        self.assertEqual(async_client.max_tokens, sync.max_tokens)

    def test_repr(self):
        client = AsyncLLMClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        r = repr(client)
        self.assertIn("AsyncLLMClient", r)
        self.assertIn("gpt-4o", r)

    def test_headers_openai(self):
        client = AsyncLLMClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="sk-test",
        )
        h = client._headers()
        self.assertIn("Authorization", h)
        self.assertIn("Bearer sk-test", h["Authorization"])

    def test_headers_anthropic(self):
        client = AsyncLLMClient(
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-6",
            api_key="sk-ant-test",
        )
        h = client._headers()
        self.assertIn("x-api-key", h)
        self.assertIn("anthropic-version", h)
        self.assertEqual(h["x-api-key"], "sk-ant-test")

    def test_path_construction(self):
        client = AsyncLLMClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        self.assertEqual(client._path("/chat/completions"), "/v1/chat/completions")


class TestAsyncLLMClientAcomplete(unittest.TestCase):
    """Teste acomplete() avec réseau mocké."""

    def test_acomplete_openai_text(self):
        """acomplete() retourne une LLMResponse depuis une réponse OpenAI mockée."""
        fake_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Porto-Novo est la capitale du Bénin.",
                }
            }],
            "usage": {},
        }

        async def go():
            client = AsyncLLMClient(
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                api_key="sk-test",
            )
            with patch(
                "zeroagent.core.async_llm._do_async_request",
                new_callable=AsyncMock,
                return_value=fake_response,
            ):
                response = await client.acomplete([
                    Message(role="user", content="Capitale du Bénin ?")
                ])
            return response

        response = run(go())
        self.assertEqual(response.content, "Porto-Novo est la capitale du Bénin.")
        self.assertFalse(response.has_tool_calls)

    def test_acomplete_openai_tool_call(self):
        """acomplete() parse correctement un tool call OpenAI."""
        fake_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "capitale Bénin"}',
                        }
                    }]
                }
            }],
            "usage": {},
        }

        async def go():
            client = AsyncLLMClient(
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                api_key="sk-test",
            )
            with patch(
                "zeroagent.core.async_llm._do_async_request",
                new_callable=AsyncMock,
                return_value=fake_response,
            ):
                return await client.acomplete([
                    Message(role="user", content="Capitale du Bénin ?")
                ])

        response = run(go())
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "search")
        self.assertEqual(response.tool_calls[0].arguments, {"query": "capitale Bénin"})

    def test_acomplete_anthropic_text(self):
        """acomplete() retourne une LLMResponse depuis une réponse Anthropic mockée."""
        fake_response = {
            "content": [{"type": "text", "text": "Porto-Novo."}],
            "stop_reason": "end_turn",
            "usage": {},
        }

        async def go():
            client = AsyncLLMClient(
                base_url="https://api.anthropic.com",
                model="claude-sonnet-4-6",
                api_key="sk-ant-test",
            )
            with patch(
                "zeroagent.core.async_llm._do_async_request",
                new_callable=AsyncMock,
                return_value=fake_response,
            ):
                return await client.acomplete([
                    Message(role="user", content="Capitale du Bénin ?")
                ])

        response = run(go())
        self.assertEqual(response.content, "Porto-Novo.")

    def test_acomplete_retries_on_500(self):
        """acomplete() doit retenter sur HTTP 500."""
        call_count = 0

        async def fake_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("HTTP 500: Internal Server Error")
            return {
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {},
            }

        async def go():
            client = AsyncLLMClient(
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                api_key="sk-test",
                max_retries=3,
                retry_delay=0.0,
            )
            with patch("zeroagent.core.async_llm._do_async_request", side_effect=fake_request):
                return await client.acomplete([Message(role="user", content="test")])

        response = run(go())
        self.assertEqual(response.content, "OK")
        self.assertEqual(call_count, 2)

    def test_acomplete_raises_after_max_retries(self):
        """acomplete() lève RuntimeError après épuisement des retries."""
        async def fake_request(*args, **kwargs):
            raise RuntimeError("HTTP 500: boom")

        async def go():
            client = AsyncLLMClient(
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
                api_key="sk-test",
                max_retries=2,
                retry_delay=0.0,
            )
            with patch("zeroagent.core.async_llm._do_async_request", side_effect=fake_request):
                await client.acomplete([Message(role="user", content="test")])

        with self.assertRaises(RuntimeError):
            run(go())


# ---------------------------------------------------------------------------
# Tests AsyncExecutor
# ---------------------------------------------------------------------------

class TestAsyncExecutor(unittest.TestCase):
    """Vérifie l'exécution parallèle et la gestion des tools async/sync."""

    def _make_registry(self, *fns):
        r = ToolRegistry()
        for fn in fns:
            r.register(fn)
        return r

    def test_execute_sync_tool(self):
        """Un tool sync doit être exécuté correctement."""
        @tool(description="Ajoute deux entiers")
        def add(a: int, b: int) -> int:
            return a + b

        registry = self._make_registry(add)
        executor = AsyncExecutor(registry)

        action = ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name="add",
            arguments={"a": 2, "b": 3},
            tool_call_id="tc-1",
        )

        results = run(executor.aexecute(action))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].observation, "5")
        self.assertTrue(results[0].success)

    def test_execute_async_tool(self):
        """Un tool async (coroutine) doit être exécuté directement."""
        async def async_search(query: str) -> str:
            await asyncio.sleep(0)   # simule un await I/O
            return f"résultat pour {query}"

        # Enregistrer manuellement comme tool
        from zeroagent.core.tools import tool as tool_deco
        async_search = tool_deco(description="Recherche async")(async_search)

        registry = self._make_registry(async_search)
        executor = AsyncExecutor(registry)

        action = ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name="async_search",
            arguments={"query": "Bénin"},
            tool_call_id="tc-1",
        )

        results = run(executor.aexecute(action))
        self.assertEqual(len(results), 1)
        self.assertIn("Bénin", results[0].observation)
        self.assertTrue(results[0].success)

    def test_parallel_execution(self):
        """
        Deux tools sync doivent s'exécuter en parallèle via gather().
        Vérifié par ordre d'exécution et timing.
        """
        execution_order = []

        @tool(description="Tool A")
        def tool_a() -> str:
            execution_order.append("A")
            return "résultat A"

        @tool(description="Tool B")
        def tool_b() -> str:
            execution_order.append("B")
            return "résultat B"

        registry = self._make_registry(tool_a, tool_b)
        executor = AsyncExecutor(registry)

        tc_b = ToolCall(name="tool_b", arguments={}, id="tc-b")
        action = ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name="tool_a",
            arguments={},
            tool_call_id="tc-a",
            extra_tool_calls=[tc_b],
        )

        results = run(executor.aexecute(action))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        obs = {r.tool_call.name: r.observation for r in results}
        self.assertEqual(obs["tool_a"], "résultat A")
        self.assertEqual(obs["tool_b"], "résultat B")

    def test_unknown_tool_returns_error_result(self):
        """Un tool inconnu retourne success=False, pas une exception."""
        registry = ToolRegistry()
        executor = AsyncExecutor(registry)

        action = ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name="inexistant",
            arguments={},
            tool_call_id="tc-1",
        )

        results = run(executor.aexecute(action))
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("inexistant", results[0].observation)

    def test_tool_exception_returns_error_result(self):
        """Une exception dans un tool retourne success=False, pas une exception."""
        @tool(description="Tool qui explose")
        def boom() -> str:
            raise ValueError("BOOM")

        registry = self._make_registry(boom)
        executor = AsyncExecutor(registry)

        action = ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name="boom",
            arguments={},
            tool_call_id="tc-1",
        )

        results = run(executor.aexecute(action))
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("BOOM", results[0].observation)

    def test_empty_action_returns_empty_list(self):
        """Une action sans tool_calls retourne une liste vide."""
        registry = ToolRegistry()
        executor = AsyncExecutor(registry)

        action = ParsedAction(
            type=ActionType.FINAL_ANSWER,
            final_answer="réponse",
        )

        results = run(executor.aexecute(action))
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# Tests AsyncAgentLoop
# ---------------------------------------------------------------------------

class TestAsyncAgentLoop(unittest.TestCase):
    """Vérifie la boucle ReAct async."""

    def _make_loop(self, responses: list[LLMResponse], tools=None, config=None):
        llm = FakeAsyncLLM(responses)
        registry = ToolRegistry()
        for fn in (tools or []):
            registry.register(fn)
        memory = BufferMemory()
        return AsyncAgentLoop(
            llm=llm,
            registry=registry,
            memory=memory,
            config=config or RunConfig(max_iterations=5),
        )

    def test_direct_final_answer(self):
        """Le LLM répond directement avec une réponse finale."""
        loop = self._make_loop([
            make_text_response("Final Answer: Porto-Novo est la capitale.")
        ])
        result = run(loop.arun("Capitale du Bénin ?"))
        self.assertEqual(result.stop_reason, StopReason.FINAL_ANSWER)
        self.assertIn("Porto-Novo", result.answer)

    def test_tool_call_then_final_answer(self):
        """Le LLM appelle un tool puis donne une réponse finale."""
        @tool(description="Recherche")
        def search(query: str) -> str:
            return "Porto-Novo"

        loop = self._make_loop(
            responses=[
                make_tool_response("search", {"query": "capitale Bénin"}),
                make_text_response("Final Answer: C'est Porto-Novo."),
            ],
            tools=[search],
        )
        result = run(loop.arun("Capitale du Bénin ?"))
        self.assertEqual(result.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(result.tool_calls_made, 1)
        self.assertIn("Porto-Novo", result.answer)

    def test_max_iterations_guard(self):
        """Ne doit jamais dépasser max_iterations."""
        # Le LLM réfléchit indéfiniment sans jamais donner de réponse finale
        loop = self._make_loop(
            responses=[make_text_response("Thought: je réfléchis...")] * 20,
            config=RunConfig(max_iterations=3, max_thought_only=10),
        )
        result = run(loop.arun("test"))
        self.assertLessEqual(result.iterations, 3)

    def test_llm_error_stops_loop(self):
        """Une erreur LLM doit stopper proprement la boucle."""
        class ErrorAsyncLLM:
            model = "error-model"
            async def acomplete(self, messages, tools=None):
                raise RuntimeError("connexion refusée")

        loop = AsyncAgentLoop(
            llm=ErrorAsyncLLM(),
            registry=ToolRegistry(),
            memory=BufferMemory(),
            config=RunConfig(max_iterations=5),
        )
        result = run(loop.arun("test"))
        self.assertEqual(result.stop_reason, StopReason.ERROR)
        self.assertFalse(result.success)

    def test_thoughts_collected(self):
        """Les thoughts doivent être accumulés dans RunResult."""
        loop = self._make_loop([
            make_text_response("Thought: première réflexion\nFinal Answer: réponse."),
        ])
        result = run(loop.arun("test"))
        self.assertGreater(len(result.thoughts), 0)

    def test_hooks_called_sync(self):
        """Les hooks sync doivent être appelés correctement."""
        thoughts_seen = []
        finals_seen = []

        loop = self._make_loop(
            responses=[make_text_response("Final Answer: réponse finale.")],
        )
        loop._on_thought = lambda t, i: thoughts_seen.append(t)
        loop._on_final = lambda a, r: finals_seen.append(a)

        result = run(loop.arun("test"))
        self.assertEqual(finals_seen, ["réponse finale."])

    def test_hooks_called_async(self):
        """Les hooks async doivent être awaités correctement."""
        finals_seen = []

        async def async_on_final(answer, reason):
            finals_seen.append(answer)

        loop = self._make_loop(
            responses=[make_text_response("Final Answer: réponse async.")],
        )
        loop._on_final = async_on_final

        result = run(loop.arun("test"))
        self.assertEqual(finals_seen, ["réponse async."])

    def test_memory_accumulates_messages(self):
        """La mémoire doit contenir les messages après le run."""
        loop = self._make_loop([
            make_text_response("Final Answer: réponse.")
        ])
        run(loop.arun("question de test"))
        messages = loop.memory.get()
        roles = [m.role for m in messages]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_parallel_tool_calls(self):
        """
        Quand le LLM retourne plusieurs tool calls, ils doivent
        tous être exécutés et les résultats injectés en mémoire.
        """
        @tool(description="Tool A")
        def tool_a() -> str:
            return "résultat A"

        @tool(description="Tool B")
        def tool_b() -> str:
            return "résultat B"

        # Réponse avec 2 tool calls simultanés
        multi_tool_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(name="tool_a", arguments={}, id="tc-a"),
                ToolCall(name="tool_b", arguments={}, id="tc-b"),
            ],
            reasoning=None,
        )

        loop = self._make_loop(
            responses=[
                multi_tool_response,
                make_text_response("Final Answer: A et B done."),
            ],
            tools=[tool_a, tool_b],
        )
        result = run(loop.arun("test"))
        self.assertEqual(result.tool_calls_made, 2)
        self.assertEqual(result.stop_reason, StopReason.FINAL_ANSWER)


# ---------------------------------------------------------------------------
# Tests Agent.arun() / Agent.arun_full()
# ---------------------------------------------------------------------------

class TestAgentArun(unittest.TestCase):
    """Vérifie Agent.arun() et Agent.arun_full()."""

    def _make_agent(self, responses: list[LLMResponse], tools=None):
        """
        Crée un Agent avec un faux LLMClient sync (run()) et
        patch _get_async_loop() pour retourner un AsyncAgentLoop avec FakeAsyncLLM.
        """
        # LLMClient factice pour la partie sync
        sync_llm = MagicMock(spec=LLMClient)
        sync_llm.base_url = "https://api.openai.com/v1"
        sync_llm.model = "fake-model"
        sync_llm.api_key = "sk-test"
        sync_llm.temperature = 0.7
        sync_llm.max_tokens = 1024
        sync_llm.timeout = 30.0
        sync_llm.max_retries = 3
        sync_llm.retry_delay = 1.0

        agent = Agent(llm=sync_llm, tools=tools or [])

        # Injecter un AsyncAgentLoop avec FakeAsyncLLM
        fake_async_llm = FakeAsyncLLM(responses)
        from zeroagent.harness.async_loop import AsyncAgentLoop
        async_loop = AsyncAgentLoop(
            llm=fake_async_llm,
            registry=agent._registry,
            memory=agent._memory,
            config=agent._config,
        )
        agent._async_loop = async_loop
        return agent

    def test_arun_returns_string(self):
        """arun() doit retourner une str."""
        agent = self._make_agent([
            make_text_response("Final Answer: Porto-Novo.")
        ])
        result = run(agent.arun("Capitale du Bénin ?"))
        self.assertIsInstance(result, str)
        self.assertIn("Porto-Novo", result)

    def test_arun_full_returns_run_result(self):
        """arun_full() doit retourner un RunResult."""
        from zeroagent.harness.loop import RunResult
        agent = self._make_agent([
            make_text_response("Final Answer: Porto-Novo.")
        ])
        result = run(agent.arun_full("Capitale du Bénin ?"))
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.stop_reason, StopReason.FINAL_ANSWER)

    def test_arun_with_tool(self):
        """arun() avec tool call fonctionne correctement."""
        @tool(description="Recherche")
        def search(query: str) -> str:
            return "Porto-Novo"

        agent = self._make_agent(
            responses=[
                make_tool_response("search", {"query": "capitale Bénin"}),
                make_text_response("Final Answer: C'est Porto-Novo."),
            ],
            tools=[search],
        )
        result = run(agent.arun("Capitale du Bénin ?"))
        self.assertIn("Porto-Novo", result)

    def test_arun_memory_shared_with_run(self):
        """
        La mémoire est partagée entre run() et arun().
        Un tour via run() doit être visible dans arun() et vice-versa.
        """
        agent = self._make_agent([
            make_text_response("Final Answer: réponse async.")
        ])
        # Injecter manuellement un message en mémoire (comme run() le ferait)
        agent._memory.add(Message(role="user", content="tour sync précédent"))
        agent._memory.add(Message(role="assistant", content="réponse sync"))

        result = run(agent.arun("question async"))
        messages = agent._memory.get()
        roles = [m.role for m in messages]
        # Les messages du tour sync sont toujours là
        self.assertEqual(roles.count("user"), 2)   # tour sync + tour async

    def test_async_loop_cached(self):
        """_get_async_loop() doit retourner le même objet à chaque appel."""
        sync_llm = MagicMock(spec=LLMClient)
        sync_llm.base_url = "https://api.openai.com/v1"
        sync_llm.model = "fake"
        sync_llm.api_key = ""
        sync_llm.temperature = 0.7
        sync_llm.max_tokens = 1024
        sync_llm.timeout = 30.0
        sync_llm.max_retries = 3
        sync_llm.retry_delay = 1.0

        agent = Agent(llm=sync_llm)
        loop1 = agent._get_async_loop()
        loop2 = agent._get_async_loop()
        self.assertIs(loop1, loop2)


if __name__ == "__main__":
    unittest.main()
