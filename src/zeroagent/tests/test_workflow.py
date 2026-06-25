"""
tests/test_workflow.py — Tests pour State, Node, Graph.
Zéro dépendance externe. Zéro appel LLM réel.
"""

from __future__ import annotations

import asyncio
import unittest

from zeroagent.workflow.state import State
from zeroagent.workflow.node import Node
from zeroagent.workflow.graph import Graph, GraphResult, Edge, ConditionalEdge


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests State
# ---------------------------------------------------------------------------

class TestState(unittest.TestCase):

    def test_init_empty(self):
        s = State()
        self.assertEqual(len(s), 0)

    def test_init_with_data(self):
        s = State({"a": 1, "b": "hello"})
        self.assertEqual(s["a"], 1)
        self.assertEqual(s["b"], "hello")

    def test_setitem_getitem(self):
        s = State()
        s["key"] = "value"
        self.assertEqual(s["key"], "value")

    def test_contains(self):
        s = State({"x": 1})
        self.assertIn("x", s)
        self.assertNotIn("y", s)

    def test_get_default(self):
        s = State({"x": 1})
        self.assertEqual(s.get("x"), 1)
        self.assertIsNone(s.get("missing"))
        self.assertEqual(s.get("missing", 42), 42)

    def test_update(self):
        s = State({"a": 1})
        s.update({"b": 2, "c": 3})
        self.assertEqual(s["b"], 2)
        self.assertEqual(s["c"], 3)

    def test_merge_returns_self(self):
        s = State({"a": 1})
        result = s.merge({"b": 2})
        self.assertIs(result, s)
        self.assertEqual(s["b"], 2)

    def test_copy_is_deep(self):
        s = State({"list": [1, 2, 3]})
        s2 = s.copy()
        s2["list"].append(4)
        self.assertEqual(s["list"], [1, 2, 3])   # original non modifié

    def test_copy_same_type(self):
        class MyState(State):
            pass
        s = MyState({"x": 1})
        s2 = s.copy()
        self.assertIsInstance(s2, MyState)

    def test_to_dict(self):
        s = State({"a": 1, "b": 2})
        d = s.to_dict()
        self.assertEqual(d, {"a": 1, "b": 2})

    def test_to_json(self):
        s = State({"a": 1})
        j = s.to_json()
        self.assertIn('"a"', j)
        self.assertIn("1", j)

    def test_from_dict(self):
        s = State.from_dict({"x": 42})
        self.assertEqual(s["x"], 42)

    def test_eq_state(self):
        s1 = State({"a": 1})
        s2 = State({"a": 1})
        self.assertEqual(s1, s2)

    def test_eq_dict(self):
        s = State({"a": 1})
        self.assertEqual(s, {"a": 1})

    def test_repr(self):
        s = State({"a": 1, "b": 2})
        r = repr(s)
        self.assertIn("State", r)

    def test_iter(self):
        s = State({"a": 1, "b": 2})
        keys = list(s)
        self.assertIn("a", keys)
        self.assertIn("b", keys)

    def test_len(self):
        s = State({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(s), 3)

    def test_delitem(self):
        s = State({"a": 1, "b": 2})
        del s["a"]
        self.assertNotIn("a", s)

    def test_items_keys_values(self):
        s = State({"a": 1})
        self.assertIn("a", s.keys())
        self.assertIn(1, s.values())
        self.assertIn(("a", 1), s.items())


class TestStateSubclass(unittest.TestCase):
    """Sous-classes avec schéma et valeurs par défaut."""

    def test_defaults_injected(self):
        class PipelineState(State):
            result: str = ""
            retries: int = 0

        s = PipelineState({"query": "test"})
        self.assertEqual(s["result"], "")
        self.assertEqual(s["retries"], 0)
        self.assertEqual(s["query"], "test")

    def test_provided_values_override_defaults(self):
        class PipelineState(State):
            result: str = ""

        s = PipelineState({"result": "already done"})
        self.assertEqual(s["result"], "already done")

    def test_validate_required_missing(self):
        class StrictState(State):
            query: str       # requis — pas de défaut

        s = StrictState({})
        errors = s.validate()
        self.assertTrue(any("query" in e for e in errors))

    def test_validate_ok(self):
        class StrictState(State):
            query: str

        s = StrictState({"query": "test"})
        errors = s.validate()
        self.assertEqual(errors, [])

    def test_validate_base_state_always_ok(self):
        s = State({"anything": "goes"})
        self.assertEqual(s.validate(), [])


# ---------------------------------------------------------------------------
# Tests Node
# ---------------------------------------------------------------------------

class TestNode(unittest.TestCase):

    def _make_node(self, name="step", fn=None, **kwargs):
        if fn is None:
            def fn(state: State) -> State:
                state["visited"] = True
                return state
        return Node(name, fn=fn, **kwargs)

    def test_init_basic(self):
        node = self._make_node()
        self.assertEqual(node.name, "step")
        self.assertFalse(node._is_async)

    def test_repr(self):
        node = self._make_node()
        r = repr(node)
        self.assertIn("step", r)
        self.assertIn("Node", r)

    def test_eq_and_hash(self):
        def fn(s): return s
        n1 = Node("a", fn=fn)
        n2 = Node("a", fn=fn)
        n3 = Node("b", fn=fn)
        self.assertEqual(n1, n2)
        self.assertNotEqual(n1, n3)
        self.assertEqual(hash(n1), hash(n2))

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            Node("", fn=lambda s: s)

    def test_invalid_retries_raises(self):
        with self.assertRaises(ValueError):
            Node("step", fn=lambda s: s, retries=-1)

    def test_run_sync(self):
        def fn(state: State) -> State:
            state["done"] = True
            return state

        node = Node("step", fn=fn)
        result = node.run(State({}))
        self.assertTrue(result["done"])

    def test_run_returns_state(self):
        node = self._make_node()
        result = node.run(State({}))
        self.assertIsInstance(result, State)

    def test_run_fn_not_returning_state_raises(self):
        def bad_fn(state):
            return "not a state"
        node = Node("bad", fn=bad_fn)
        with self.assertRaises(TypeError):
            node.run(State({}))

    def test_run_with_retries(self):
        call_count = [0]

        def flaky(state: State) -> State:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("fail")
            state["done"] = True
            return state

        node = Node("step", fn=flaky, retries=3)
        result = node.run(State({}))
        self.assertTrue(result["done"])
        self.assertEqual(call_count[0], 3)

    def test_run_exhausted_retries_raises(self):
        def always_fail(state):
            raise RuntimeError("always")
        node = Node("step", fn=always_fail, retries=2)
        with self.assertRaises(RuntimeError):
            node.run(State({}))

    def test_arun_async_fn(self):
        async def async_fn(state: State) -> State:
            state["async"] = True
            return state

        node = Node("async_step", fn=async_fn)
        self.assertTrue(node._is_async)
        result = run(node.arun(State({})))
        self.assertTrue(result["async"])

    def test_arun_sync_fn(self):
        """Un nœud sync peut être awaité — run_in_executor."""
        def sync_fn(state: State) -> State:
            state["sync_in_async"] = True
            return state

        node = Node("sync_step", fn=sync_fn)
        result = run(node.arun(State({})))
        self.assertTrue(result["sync_in_async"])

    def test_node_with_agent_duck_type(self):
        """Un objet avec .run() est traité comme un Agent."""
        class FakeAgent:
            def run(self, input_str: str) -> str:
                return f"réponse à: {input_str}"

        agent = FakeAgent()
        node = Node("agent_node", fn=agent)
        result = node.run(State({"input": "question"}))
        self.assertIn("réponse à", result["output"])

    def test_async_node_with_agent_arun(self):
        """Un objet avec .arun() est détecté comme async."""
        class FakeAsyncAgent:
            async def arun(self, input_str: str) -> str:
                return f"async réponse à: {input_str}"

        agent = FakeAsyncAgent()
        node = Node("async_agent", fn=agent)
        self.assertTrue(node._is_async)
        result = run(node.arun(State({"input": "question"})))
        self.assertIn("async réponse à", result["output"])


# ---------------------------------------------------------------------------
# Tests Graph
# ---------------------------------------------------------------------------

class TestGraphConstruction(unittest.TestCase):

    def _fn(self, key):
        def fn(state: State) -> State:
            state[key] = True
            return state
        return fn

    def test_add_node(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        self.assertIn("a", g._nodes)

    def test_add_node_duplicate_raises(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        with self.assertRaises(ValueError):
            g.add_node(Node("a", fn=self._fn("a")))

    def test_add_edge(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.add_node(Node("b", fn=self._fn("b")))
        g.add_edge("a", "b")
        self.assertEqual(len(g._edges), 1)

    def test_add_edge_unknown_node_raises(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        with self.assertRaises(ValueError):
            g.add_edge("a", "unknown")

    def test_add_edge_cycle_raises(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.add_node(Node("b", fn=self._fn("b")))
        g.add_edge("a", "b")
        with self.assertRaises(ValueError):
            g.add_edge("b", "a")

    def test_chaining(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.add_node(Node("b", fn=self._fn("b")))
        result = g.add_edge("a", "b")
        self.assertIs(result, g)

    def test_set_entry_unknown_raises(self):
        g = Graph()
        with self.assertRaises(ValueError):
            g.set_entry("unknown")

    def test_set_finish_unknown_raises(self):
        g = Graph()
        with self.assertRaises(ValueError):
            g.set_finish("unknown")

    def test_add_conditional_edge(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.add_node(Node("b", fn=self._fn("b")))
        g.add_node(Node("c", fn=self._fn("c")))
        g.add_conditional_edge("a", lambda s: "ok", {"ok": "b", "err": "c"})
        self.assertEqual(len(g._conditional_edges), 1)

    def test_repr(self):
        g = Graph(name="test_graph")
        r = repr(g)
        self.assertIn("test_graph", r)
        self.assertIn("Graph", r)


class TestGraphRun(unittest.TestCase):

    def _fn(self, key, value=True):
        def fn(state: State) -> State:
            state[key] = value
            return state
        return fn

    def _simple_graph(self):
        g = Graph()
        g.add_node(Node("fetch", fn=self._fn("fetched")))
        g.add_node(Node("process", fn=self._fn("processed")))
        g.add_node(Node("output", fn=self._fn("done")))
        g.add_edge("fetch", "process")
        g.add_edge("process", "output")
        g.set_entry("fetch")
        g.set_finish("output")
        return g

    def test_run_linear_graph(self):
        g = self._simple_graph()
        result = g.run(State({}))
        self.assertTrue(result.success)
        self.assertTrue(result.state["fetched"])
        self.assertTrue(result.state["processed"])
        self.assertTrue(result.state["done"])

    def test_run_executed_order(self):
        g = self._simple_graph()
        result = g.run(State({}))
        self.assertEqual(result.executed, ["fetch", "process", "output"])

    def test_run_no_entry_raises(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.set_finish("a")
        with self.assertRaises(RuntimeError):
            g.run(State({}))

    def test_run_no_finish_raises(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.set_entry("a")
        with self.assertRaises(RuntimeError):
            g.run(State({}))

    def test_run_node_error_returns_failure(self):
        def bad(state):
            raise ValueError("boom")

        g = Graph()
        g.add_node(Node("a", fn=bad))
        g.add_node(Node("b", fn=self._fn("b")))
        g.add_edge("a", "b")
        g.set_entry("a")
        g.set_finish("b")
        result = g.run(State({}))
        self.assertFalse(result.success)
        self.assertIn("boom", result.error)

    def test_run_state_passed_through(self):
        def step1(state: State) -> State:
            state["x"] = 10
            return state

        def step2(state: State) -> State:
            state["y"] = state["x"] * 2
            return state

        g = Graph()
        g.add_node(Node("s1", fn=step1))
        g.add_node(Node("s2", fn=step2))
        g.add_edge("s1", "s2")
        g.set_entry("s1")
        g.set_finish("s2")
        result = g.run(State({}))
        self.assertEqual(result.state["y"], 20)

    def test_run_conditional_edge(self):
        def route(state: State) -> str:
            return "yes" if state.get("flag") else "no"

        def yes_fn(state: State) -> State:
            state["branch"] = "yes"
            return state

        def no_fn(state: State) -> State:
            state["branch"] = "no"
            return state

        g = Graph()
        g.add_node(Node("start", fn=self._fn("started")))
        g.add_node(Node("yes", fn=yes_fn))
        g.add_node(Node("no", fn=no_fn))
        g.add_conditional_edge("start", route, {"yes": "yes", "no": "no"})
        g.set_entry("start")
        g.set_finish("yes", "no")

        result_yes = g.run(State({"flag": True}))
        self.assertEqual(result_yes.state["branch"], "yes")

        result_no = g.run(State({"flag": False}))
        self.assertEqual(result_no.state["branch"], "no")

    def test_arun_linear_graph(self):
        g = self._simple_graph()
        result = run(g.arun(State({})))
        self.assertTrue(result.success)
        self.assertTrue(result.state["done"])

    def test_arun_with_async_node(self):
        async def async_fn(state: State) -> State:
            state["async_done"] = True
            return state

        g = Graph()
        g.add_node(Node("async_step", fn=async_fn))
        g.set_entry("async_step")
        g.set_finish("async_step")
        result = run(g.arun(State({})))
        self.assertTrue(result.state["async_done"])

    def test_arun_parallel_nodes(self):
        """
        Deux nœuds sans dépendances entre eux doivent s'exécuter
        en parallèle via asyncio.gather().
        """
        executed_order = []

        async def branch_a(state: State) -> State:
            executed_order.append("a")
            state["a"] = True
            return state

        async def branch_b(state: State) -> State:
            executed_order.append("b")
            state["b"] = True
            return state

        def merge(state: State) -> State:
            state["merged"] = True
            return state

        g = Graph()
        g.add_node(Node("start", fn=lambda s: s))
        g.add_node(Node("branch_a", fn=branch_a))
        g.add_node(Node("branch_b", fn=branch_b))
        g.add_node(Node("merge", fn=merge))
        g.add_edge("start", "branch_a")
        g.add_edge("start", "branch_b")
        g.add_edge("branch_a", "merge")
        g.add_edge("branch_b", "merge")
        g.set_entry("start")
        g.set_finish("merge")

        result = run(g.arun(State({})))
        self.assertTrue(result.state.get("a"))
        self.assertTrue(result.state.get("b"))
        self.assertTrue(result.state.get("merged"))

    def test_multiple_finish_nodes(self):
        g = Graph()
        g.add_node(Node("a", fn=self._fn("a")))
        g.add_node(Node("b", fn=self._fn("b")))
        g.add_node(Node("c", fn=self._fn("c")))
        g.add_edge("a", "b")
        g.set_entry("a")
        g.set_finish("b", "c")
        # S'arrête à "b" car c'est le premier finish atteint
        result = g.run(State({}))
        self.assertTrue(result.success)
        self.assertIn("b", result.executed)

    def test_graph_result_repr(self):
        r = GraphResult(state=State({}), executed=["a", "b"], success=True)
        s = repr(r)
        self.assertIn("GraphResult", s)


class TestGraphVisualize(unittest.TestCase):

    def test_visualize_empty(self):
        g = Graph(name="empty")
        v = g.visualize()
        self.assertIn("empty", v)

    def test_visualize_linear(self):
        g = Graph(name="pipeline")

        def fn(s): return s
        g.add_node(Node("fetch", fn=fn))
        g.add_node(Node("process", fn=fn))
        g.set_entry("fetch")
        g.set_finish("process")
        g.add_edge("fetch", "process")
        v = g.visualize()
        self.assertIn("fetch", v)
        self.assertIn("process", v)
        self.assertIn("entry", v)
        self.assertIn("finish", v)


# ---------------------------------------------------------------------------
# Tests imports publics workflow
# ---------------------------------------------------------------------------

class TestWorkflowPublicImports(unittest.TestCase):

    def test_from_workflow(self):
        from zeroagent.workflow import Graph, Node, State, GraphResult
        from zeroagent.workflow import Edge, ConditionalEdge
        self.assertTrue(True)

    def test_from_zeroagent(self):
        from zeroagent import Graph, Node, State, GraphResult
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
