"""
tests/test_harness.py

Tests unitaires pour tout le harness (parser, executor, loop, agent, memory).
Zéro dépendance externe : unittest + unittest.mock uniquement.
Le réseau n'est jamais appelé : LLMClient est mocké.
"""

import unittest
from unittest.mock import MagicMock, patch

from zeroagent.core.llm import LLMClient, LLMResponse, Message, ToolCall
from zeroagent.core.memory import BufferMemory, WindowMemory
from zeroagent.core.tools import ToolRegistry, tool
from zeroagent.harness.agent import Agent
from zeroagent.harness.executor import Executor, ExecutionResult
from zeroagent.harness.loop import AgentLoop, RunConfig, RunResult, StopReason
from zeroagent.harness.memory import SummaryMemory
from zeroagent.harness.parser import (
    ActionType,
    ParsedAction,
    parse_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    reasoning: str | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        model="test-model",
        reasoning=reasoning,
    )


def _mock_llm(*responses: LLMResponse) -> MagicMock:
    """Crée un LLMClient mock qui retourne les réponses dans l'ordre."""
    mock = MagicMock(spec=LLMClient)
    mock.model = "test-model"
    mock._provider = "openai"
    mock.complete.side_effect = list(responses)
    return mock


def _make_registry(*fns) -> ToolRegistry:
    r = ToolRegistry()
    for fn in fns:
        r.register(fn)
    return r


# ---------------------------------------------------------------------------
# Tests : Parser — mode structuré
# ---------------------------------------------------------------------------

class TestParserStructured(unittest.TestCase):

    def test_single_tool_call(self):
        tc = ToolCall(id="call_001", name="search", arguments={"query": "Bénin"})
        response = _llm_response(tool_calls=[tc])
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.TOOL_CALL)
        self.assertEqual(action.tool_name, "search")
        self.assertEqual(action.arguments, {"query": "Bénin"})
        self.assertEqual(action.tool_call_id, "call_001")

    def test_tool_call_with_reasoning(self):
        tc = ToolCall(id="call_001", name="search", arguments={"query": "test"})
        response = _llm_response(tool_calls=[tc], reasoning="Je dois chercher...")
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.TOOL_CALL)
        self.assertEqual(action.thought, "Je dois chercher...")

    def test_multiple_tool_calls_takes_first(self):
        tcs = [
            ToolCall(id="c1", name="search", arguments={"query": "A"}),
            ToolCall(id="c2", name="calc", arguments={"expr": "1+1"}),
        ]
        response = _llm_response(tool_calls=tcs)
        action = parse_response(response)

        self.assertEqual(action.tool_name, "search")
        self.assertEqual(action.tool_call_id, "c1")


# ---------------------------------------------------------------------------
# Tests : Parser — mode ReAct texte libre
# ---------------------------------------------------------------------------

class TestParserReAct(unittest.TestCase):

    def test_final_answer(self):
        response = _llm_response(content="Final Answer: Porto-Novo est la capitale.")
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.FINAL_ANSWER)
        self.assertEqual(action.final_answer, "Porto-Novo est la capitale.")

    def test_thought_and_action(self):
        text = (
            "Thought: je dois chercher la capitale\n"
            "Action: search\n"
            "Action Input: {\"query\": \"capitale Bénin\"}"
        )
        response = _llm_response(content=text)
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.TOOL_CALL)
        self.assertEqual(action.tool_name, "search")
        self.assertEqual(action.arguments.get("query"), "capitale Bénin")
        self.assertIn("capitale", action.thought)

    def test_thought_and_final_answer(self):
        text = (
            "Thought: je connais déjà la réponse\n"
            "Final Answer: C'est Porto-Novo."
        )
        response = _llm_response(content=text)
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.FINAL_ANSWER)
        self.assertEqual(action.final_answer, "C'est Porto-Novo.")
        self.assertIn("connais", action.thought)

    def test_direct_response_no_format(self):
        """Réponse directe sans format ReAct → final answer."""
        response = _llm_response(content="Porto-Novo est la capitale du Bénin.")
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.FINAL_ANSWER)
        self.assertEqual(action.final_answer, "Porto-Novo est la capitale du Bénin.")

    def test_empty_response(self):
        """Réponse vide → thought_only."""
        response = _llm_response(content="")
        action = parse_response(response)
        self.assertEqual(action.type, ActionType.THOUGHT_ONLY)

    def test_action_input_plain_string(self):
        """Action Input non-JSON → wrap dans {"input": ...}."""
        text = "Action: search\nAction Input: capitale du Bénin"
        response = _llm_response(content=text)
        action = parse_response(response)

        self.assertEqual(action.type, ActionType.TOOL_CALL)
        self.assertIn("input", action.arguments)

    def test_case_insensitive(self):
        """Les mots-clés ReAct sont case-insensitive."""
        text = "THOUGHT: test\nFINAL ANSWER: réponse"
        response = _llm_response(content=text)
        action = parse_response(response)
        self.assertEqual(action.type, ActionType.FINAL_ANSWER)


# ---------------------------------------------------------------------------
# Tests : Executor
# ---------------------------------------------------------------------------

class TestExecutor(unittest.TestCase):

    def setUp(self):
        @tool(description="Recherche")
        def search(query: str) -> str:
            return f"Résultat: {query}"

        @tool(description="Erreur volontaire")
        def broken(x: str) -> str:
            raise ValueError("Je suis cassé")

        self.registry = _make_registry(search, broken)
        self.executor = Executor(self.registry)

    def _action(self, name: str, args: dict, call_id: str = "c1") -> ParsedAction:
        return ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name=name,
            arguments=args,
            tool_call_id=call_id,
        )

    def test_execute_success(self):
        action = self._action("search", {"query": "Bénin"})
        results = self.executor.execute(action)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertIn("Bénin", results[0].observation)

    def test_execute_unknown_tool(self):
        action = self._action("nonexistent", {})
        results = self.executor.execute(action)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("introuvable", results[0].observation)

    def test_execute_tool_raises(self):
        action = self._action("broken", {"x": "test"})
        results = self.executor.execute(action)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("Erreur", results[0].observation)

    def test_execute_non_tool_call_returns_empty(self):
        action = ParsedAction(type=ActionType.FINAL_ANSWER, final_answer="test")
        results = self.executor.execute(action)
        self.assertEqual(results, [])

    def test_execute_result_always_str(self):
        action = self._action("search", {"query": "test"})
        results = self.executor.execute(action)
        self.assertIsInstance(results[0].observation, str)


# ---------------------------------------------------------------------------
# Tests : AgentLoop
# ---------------------------------------------------------------------------

class TestAgentLoop(unittest.TestCase):

    def setUp(self):
        @tool(description="Recherche")
        def search(query: str) -> str:
            return "Porto-Novo est la capitale du Bénin."

        self.registry = _make_registry(search)

    def _make_loop(self, llm, **kwargs) -> AgentLoop:
        return AgentLoop(
            llm=llm,
            registry=self.registry,
            memory=BufferMemory(),
            config=RunConfig(max_iterations=5, verbose=False, **kwargs),
        )

    def test_direct_final_answer(self):
        """Le LLM répond directement sans tool call."""
        llm = _mock_llm(_llm_response(content="Porto-Novo est la capitale."))
        loop = self._make_loop(llm)
        result = loop.run("Capitale du Bénin ?")

        self.assertEqual(result.stop_reason, StopReason.FINAL_ANSWER)
        self.assertIn("Porto-Novo", result.answer)
        self.assertEqual(result.iterations, 1)

    def test_tool_call_then_final_answer(self):
        """Le LLM appelle un tool, puis donne une réponse finale."""
        tc = ToolCall(id="c1", name="search", arguments={"query": "capitale Bénin"})
        llm = _mock_llm(
            _llm_response(tool_calls=[tc]),
            _llm_response(content="Final Answer: Porto-Novo."),
        )
        loop = self._make_loop(llm)
        result = loop.run("Capitale du Bénin ?")

        self.assertEqual(result.stop_reason, StopReason.FINAL_ANSWER)
        self.assertEqual(result.tool_calls_made, 1)
        self.assertEqual(result.iterations, 2)

    def test_max_iterations_guard(self):
        """Ne doit jamais dépasser max_iterations."""
        tc = ToolCall(id="c1", name="search", arguments={"query": "test"})
        # LLM appelle toujours un tool, jamais de final answer
        llm = _mock_llm(*[_llm_response(tool_calls=[tc])] * 10)
        loop = self._make_loop(llm)
        result = loop.run("test")

        self.assertEqual(result.stop_reason, StopReason.MAX_ITERATIONS)
        self.assertLessEqual(result.iterations, 5)

    def test_llm_error_stops_loop(self):
        """Une erreur LLM doit stopper proprement la boucle."""
        llm = MagicMock(spec=LLMClient)
        llm.model = "test"
        llm._provider = "openai"
        llm.complete.side_effect = RuntimeError("API down")

        loop = self._make_loop(llm)
        result = loop.run("test")

        self.assertEqual(result.stop_reason, StopReason.ERROR)
        self.assertFalse(result.success)

    def test_thoughts_collected(self):
        """Les thoughts doivent être accumulés dans RunResult."""
        response = _llm_response(
            content="Thought: je réfléchis\nFinal Answer: voilà."
        )
        llm = _mock_llm(response)
        loop = self._make_loop(llm)
        result = loop.run("test")

        self.assertTrue(len(result.thoughts) > 0)

    def test_hooks_called(self):
        """Les hooks doivent être appelés aux bons moments."""
        thoughts_seen = []
        finals_seen = []

        tc = ToolCall(id="c1", name="search", arguments={"query": "test"})
        llm = _mock_llm(
            _llm_response(tool_calls=[tc]),
            _llm_response(content="Final Answer: réponse."),
        )

        loop = AgentLoop(
            llm=llm,
            registry=self.registry,
            memory=BufferMemory(),
            config=RunConfig(max_iterations=5),
            on_final=lambda ans, reason: finals_seen.append(ans),
        )
        loop.run("test")

        self.assertEqual(len(finals_seen), 1)
        self.assertEqual(finals_seen[0], "réponse.")

    def test_memory_accumulates_messages(self):
        """La mémoire doit contenir les messages après le run."""
        llm = _mock_llm(_llm_response(content="Réponse directe."))
        memory = BufferMemory()
        loop = AgentLoop(
            llm=llm,
            registry=self.registry,
            memory=memory,
            config=RunConfig(max_iterations=5),
        )
        loop.run("Question ?")

        messages = memory.get()
        roles = [m.role for m in messages]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)


# ---------------------------------------------------------------------------
# Tests : Agent (façade)
# ---------------------------------------------------------------------------

class TestAgent(unittest.TestCase):

    def setUp(self):
        @tool(description="Recherche")
        def search(query: str) -> str:
            return "Porto-Novo"

        self.search = search

    def _make_agent(self, llm, **kwargs) -> Agent:
        return Agent(
            llm=llm,
            tools=[self.search],
            system_prompt="Tu es un assistant.",
            **kwargs,
        )

    def test_run_returns_string(self):
        llm = _mock_llm(_llm_response(content="Porto-Novo est la capitale."))
        agent = self._make_agent(llm)
        result = agent.run("Capitale ?")
        self.assertIsInstance(result, str)
        self.assertIn("Porto-Novo", result)

    def test_run_full_returns_run_result(self):
        llm = _mock_llm(_llm_response(content="Porto-Novo."))
        agent = self._make_agent(llm)
        result = agent.run_full("Capitale ?")
        self.assertIsInstance(result, RunResult)

    def test_system_prompt_injected(self):
        llm = _mock_llm(_llm_response(content="ok"))
        agent = self._make_agent(llm)
        messages = agent.memory.get()
        system_msgs = [m for m in messages if m.role == "system"]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0].content, "Tu es un assistant.")

    def test_reset_clears_memory(self):
        llm = _mock_llm(
            _llm_response(content="Réponse 1."),
            _llm_response(content="Réponse 2."),
        )
        agent = self._make_agent(llm)
        agent.run("Question 1")
        agent.reset()

        messages = agent.memory.get()
        # Après reset : seulement le system prompt
        non_system = [m for m in messages if m.role != "system"]
        self.assertEqual(len(non_system), 0)

    def test_add_tool(self):
        @tool(description="Calcul")
        def calc(expr: str) -> str:
            return str(eval(expr))  # noqa

        llm = _mock_llm(_llm_response(content="ok"))
        agent = self._make_agent(llm)
        agent.add_tool(calc)

        self.assertIn("calc", agent.tools)

    def test_tools_property(self):
        llm = _mock_llm(_llm_response(content="ok"))
        agent = self._make_agent(llm)
        self.assertIn("search", agent.tools)

    def test_chaining_add_tool(self):
        @tool(description="Test")
        def t(x: str) -> str:
            return x

        llm = _mock_llm(_llm_response(content="ok"))
        agent = self._make_agent(llm)
        result = agent.add_tool(t)
        self.assertIs(result, agent)

    def test_repr(self):
        llm = _mock_llm(_llm_response(content="ok"))
        agent = self._make_agent(llm)
        r = repr(agent)
        self.assertIn("Agent", r)

    def test_verbose_mode(self):
        """verbose=True ne doit pas crasher."""
        llm = _mock_llm(_llm_response(
            content="Thought: réflexion\nFinal Answer: réponse."
        ))
        agent = self._make_agent(llm, verbose=True)
        result = agent.run("test")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Tests : SummaryMemory
# ---------------------------------------------------------------------------

class TestSummaryMemory(unittest.TestCase):

    def test_init(self):
        mem = SummaryMemory(max_messages=10)
        self.assertEqual(mem.max_messages, 10)
        self.assertFalse(mem.has_summary)

    def test_invalid_max_messages(self):
        with self.assertRaises(ValueError):
            SummaryMemory(max_messages=2)

    def test_add_and_get_within_limit(self):
        mem = SummaryMemory(max_messages=10)
        mem.add(Message(role="system", content="Système"))
        mem.add(Message(role="user", content="Question"))
        mem.add(Message(role="assistant", content="Réponse"))

        messages = mem.get()
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(len(messages), 3)

    def test_no_llm_trim_when_exceeded(self):
        """Sans LLM, trim brutal quand max_messages dépassé."""
        mem = SummaryMemory(llm=None, max_messages=4, keep_recent=2)

        for i in range(10):
            mem.add(Message(role="user", content=f"msg {i}"))

        messages = mem.get()
        # Le résumé naïf + les messages récents
        self.assertTrue(mem.has_summary)
        self.assertLessEqual(mem.total_stored, 4)

    def test_llm_summary_called_when_exceeded(self):
        """Avec LLM, le résumé est généré quand le seuil est dépassé."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model = "test"
        summary_response = LLMResponse(content="Résumé de la conversation.", model="test")
        mock_llm.complete.return_value = summary_response

        mem = SummaryMemory(llm=mock_llm, max_messages=4, keep_recent=2)
        for i in range(6):
            mem.add(Message(role="user", content=f"msg {i}"))

        self.assertTrue(mem.has_summary)
        self.assertTrue(mock_llm.complete.called)

    def test_summary_injected_in_get(self):
        """Le résumé apparaît dans get() après le seuil."""
        mem = SummaryMemory(llm=None, max_messages=4, keep_recent=2)
        for i in range(6):
            mem.add(Message(role="user", content=f"msg {i}"))

        messages = mem.get()
        contents = [m.content for m in messages]
        # Au moins un message doit contenir "Résumé" ou le résumé naïf
        self.assertTrue(any("msg" in c for c in contents))

    def test_clear_keeps_system(self):
        mem = SummaryMemory(max_messages=10)
        mem.add(Message(role="system", content="Système"))
        mem.add(Message(role="user", content="Question"))
        mem.clear()

        messages = mem.get()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "system")

    def test_to_dict_roundtrip(self):
        mem = SummaryMemory(llm=None, max_messages=4, keep_recent=2)
        mem.add(Message(role="system", content="Système"))
        mem.add(Message(role="user", content="Question"))

        d = mem.to_dict()
        restored = SummaryMemory.from_dict(d)

        self.assertEqual(restored.max_messages, 4)
        messages = restored.get()
        self.assertEqual(messages[0].role, "system")

    def test_repr(self):
        mem = SummaryMemory(max_messages=10)
        r = repr(mem)
        self.assertIn("SummaryMemory", r)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
