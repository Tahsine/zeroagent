"""
workflow/node.py — Nœud d'un Graph de workflow.

Un Node wrappe une fonction callable `fn(state: State) -> State`.
La fonction peut être :
  - une fonction Python pure
  - une coroutine async (détectée automatiquement)
  - une instance d'Agent (duck-typed : doit avoir .run() ou .arun())
  - n'importe quel callable qui accepte un State et retourne un State

Design :
  - Pas d'héritage obligatoire — duck typing + inspection
  - Le nœud ne connaît pas le Graph — découplage total
  - Les métadonnées (retries, timeout) sont optionnelles
  - Un Node peut être réutilisé dans plusieurs Graphs

Usage :
    from zeroagent.workflow import Node, State

    def fetch_data(state: State) -> State:
        state["data"] = "résultat"
        return state

    node = Node("fetch", fn=fetch_data)
    result = node.run(State({"query": "test"}))

    # Avec un Agent zeroagent
    from zeroagent import Agent
    agent = Agent(llm=llm, tools=[search])
    node = Node("research", fn=agent)
    # agent.run(state["input"]) sera appelé, résultat dans state["output"]
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from zeroagent.workflow.state import State


# ---------------------------------------------------------------------------
# Adapteur Agent → fn(State) -> State
# ---------------------------------------------------------------------------

def _wrap_agent(agent_or_fn) -> Callable[[State], State]:
    """
    Si l'objet a .run() (duck-type Agent), on crée une fn(State) -> State
    qui lit state["input"], appelle agent.run(), et écrit state["output"].

    Sinon, on retourne la fonction telle quelle.
    """
    if hasattr(agent_or_fn, "run") and not inspect.isfunction(agent_or_fn) and not inspect.isbuiltin(agent_or_fn):
        # C'est un Agent-like avec .run() — pas une fonction ordinaire
        def _agent_fn(state: State) -> State:
            user_input = state.get("input", "")
            result = agent_or_fn.run(str(user_input))
            state["output"] = result
            return state
        return _agent_fn

    if callable(agent_or_fn):
        return agent_or_fn

    raise TypeError(
        f"Node fn doit être callable ou avoir .run(). Got: {type(agent_or_fn)}"
    )


def _wrap_agent_async(agent_or_fn) -> Callable[[State], Any]:
    """
    Version async de _wrap_agent.
    Si l'objet a .arun(), on crée une coroutine fn(State) -> State.
    """
    if hasattr(agent_or_fn, "arun"):
        async def _async_agent_fn(state: State) -> State:
            user_input = state.get("input", "")
            result = await agent_or_fn.arun(str(user_input))
            state["output"] = result
            return state
        return _async_agent_fn

    if hasattr(agent_or_fn, "run"):
        async def _sync_agent_async_fn(state: State) -> State:
            user_input = state.get("input", "")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: agent_or_fn.run(str(user_input))
            )
            state["output"] = result
            return state
        return _sync_agent_async_fn

    if inspect.iscoroutinefunction(inspect.unwrap(agent_or_fn) if callable(agent_or_fn) else agent_or_fn):
        return agent_or_fn

    if callable(agent_or_fn):
        # Sync fn — on la wrappe dans run_in_executor
        async def _sync_in_async(state: State) -> State:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: agent_or_fn(state))
        return _sync_in_async

    raise TypeError(
        f"Node fn doit être callable ou avoir .arun()/.run(). Got: {type(agent_or_fn)}"
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    Unité de travail dans un Graph.

    Un Node wrappe une fonction `fn(state: State) -> State`.
    La fonction peut être sync ou async — Node s'adapte automatiquement.

    Paramètres :
        name    : identifiant unique dans le Graph (str)
        fn      : callable(State) -> State, ou Agent zeroagent
        retries : nombre de retentatives en cas d'erreur (défaut 0)
        timeout : timeout en secondes pour l'exécution async (défaut None)
        metadata: dict libre pour annotations custom

    Usage :
        # Fonction pure
        node = Node("step1", fn=my_function)

        # Agent zeroagent
        node = Node("research", fn=my_agent)

        # Async natif
        async def async_step(state: State) -> State:
            await asyncio.sleep(0.1)
            state["done"] = True
            return state

        node = Node("async_step", fn=async_step)

        # Exécution sync
        result = node.run(state)

        # Exécution async
        result = await node.arun(state)
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        retries: int = 0,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Node.name doit être une str non vide")
        if retries < 0:
            raise ValueError("Node.retries doit être >= 0")

        self.name = name
        self._fn_raw = fn
        self.retries = retries
        self.timeout = timeout
        self.metadata = metadata or {}

        # Déterminer si la fn sous-jacente est async
        raw = inspect.unwrap(fn) if callable(fn) else fn
        self._is_async = inspect.iscoroutinefunction(raw) or (
            hasattr(fn, "arun")
        )

    # ------------------------------------------------------------------
    # Exécution sync
    # ------------------------------------------------------------------

    def run(self, state: State) -> State:
        """
        Exécute le nœud de façon synchrone.
        Si la fn est async, on la lance dans asyncio.run().
        Retries automatiques en cas d'exception.
        """
        fn = _wrap_agent(self._fn_raw)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self._is_async:
                    # Fn async appelée depuis contexte sync
                    async_fn = _wrap_agent_async(self._fn_raw)
                    if self.timeout:
                        async def _with_timeout(s):
                            return await asyncio.wait_for(async_fn(s), self.timeout)
                        result = asyncio.run(_with_timeout(state))
                    else:
                        result = asyncio.run(async_fn(state))
                else:
                    result = fn(state)

                if not isinstance(result, State):
                    raise TypeError(
                        f"Node '{self.name}' doit retourner un State, got {type(result)}"
                    )
                return result
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    continue
                raise

        raise last_error  # type: ignore

    # ------------------------------------------------------------------
    # Exécution async
    # ------------------------------------------------------------------

    async def arun(self, state: State) -> State:
        """
        Exécute le nœud de façon asynchrone.
        Les fonctions sync sont wrappées dans run_in_executor.
        Retries automatiques en cas d'exception.
        """
        async_fn = _wrap_agent_async(self._fn_raw)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self.timeout:
                    result = await asyncio.wait_for(async_fn(state), self.timeout)
                else:
                    result = await async_fn(state)

                if not isinstance(result, State):
                    raise TypeError(
                        f"Node '{self.name}' doit retourner un State, got {type(result)}"
                    )
                return result
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    continue
                raise

        raise last_error  # type: ignore

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        async_tag = " [async]" if self._is_async else ""
        retry_tag = f" retries={self.retries}" if self.retries else ""
        return f"Node({self.name!r}{async_tag}{retry_tag})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)
