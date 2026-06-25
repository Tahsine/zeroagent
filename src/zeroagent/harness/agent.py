"""
harness/agent.py — Agent = LLM + Harness.

Façade publique qui assemble tous les composants du harness.
C'est le seul fichier que l'utilisateur final importe.

Usage:
    from zeroagent.harness.agent import Agent
    from zeroagent.core.llm import LLMClient
    from zeroagent.core.tools import tool

    @tool(description="Recherche des informations")
    def search(query: str) -> str:
        return f"Résultat pour : {query}"

    agent = Agent(
        llm=LLMClient(base_url="...", model="...", api_key="..."),
        tools=[search],
        system_prompt="Tu es un assistant utile.",
        max_iterations=10,
        verbose=False,
    )

    result = agent.run("Quelle est la capitale du Bénin ?")
    print(result)          # "Porto-Novo est la capitale du Bénin."
"""

from __future__ import annotations

import asyncio
from typing import Callable

from zeroagent.core.llm import LLMClient, Message
from zeroagent.core.memory import BaseMemory, BufferMemory
from zeroagent.core.tools import ToolRegistry
from zeroagent.harness.loop import AgentLoop, RunConfig, RunResult, StopReason


class Agent:
    """
    Agent IA : LLM + Harness.

    Assemble LLMClient, ToolRegistry, BaseMemory et AgentLoop
    en une interface simple à utiliser.

    Args:
        llm            : client LLM configuré
        tools          : liste de fonctions décorées avec @tool
        memory         : mémoire conversationnelle (défaut: BufferMemory)
        system_prompt  : prompt système injecté au début de chaque run
        max_iterations : nombre max d'itérations ReAct (défaut: 10)
        verbose        : affiche Thought/Action/Observation dans le terminal
        on_thought     : hook appelé à chaque Thought(thought, iteration)
        on_action      : hook appelé à chaque Action(name, args, iteration)
        on_observation : hook appelé à chaque Observation(name, obs, ok, iter)
        on_final       : hook appelé à la réponse finale(answer, stop_reason)
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Callable] | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str = "",
        max_iterations: int = 10,
        verbose: bool = False,
        on_thought: Callable | None = None,
        on_action: Callable | None = None,
        on_observation: Callable | None = None,
        on_final: Callable | None = None,
    ) -> None:
        self.llm           = llm
        self.system_prompt = system_prompt
        self.verbose       = verbose

        # Registry
        self._registry = ToolRegistry()
        for fn in (tools or []):
            self._registry.register(fn)

        # Mémoire
        self._memory = memory or BufferMemory()

        # Injecter le system prompt si fourni
        if system_prompt:
            self._memory.add(Message(role="system", content=system_prompt))

        # Loop
        self._config = RunConfig(
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self._loop = AgentLoop(
            llm=llm,
            registry=self._registry,
            memory=self._memory,
            config=self._config,
            on_thought=on_thought,
            on_action=on_action,
            on_observation=on_observation,
            on_final=on_final,
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Lance l'agent sur un input utilisateur.
        Retourne la réponse finale sous forme de string.

        La mémoire est conservée entre les appels — l'agent se souvient
        des tours précédents dans la même session.
        """
        result: RunResult = self._loop.run(user_input)
        return result.answer

    def run_full(self, user_input: str) -> RunResult:
        """
        Comme run(), mais retourne le RunResult complet :
        réponse, stop_reason, iterations, thoughts, tool_calls_made.

        Utile pour le debug, les tests, et le build in public. 😄
        """
        return self._loop.run(user_input)

    async def arun(self, user_input: str) -> str:
        """
        Version async de run().
        Retourne la réponse finale sous forme de string.

        Utilise AsyncLLMClient + AsyncAgentLoop en interne.
        Les tool calls multiples dans la même itération sont exécutés
        en parallèle via asyncio.gather().

        Usage:
            result = await agent.arun("Quelle est la capitale du Bénin ?")

        Ou dans un contexte FastAPI / Streamlit :
            answer = await agent.arun(user_message)
        """
        result = await self._get_async_loop().arun(user_input)
        return result.answer

    async def arun_full(self, user_input: str) -> RunResult:
        """
        Version async de run_full().
        Retourne le RunResult complet.

        Usage:
            result = await agent.arun_full("Calcule 42 * 7")
            print(result.answer)
            print(result.stop_reason)
            print(result.iterations)
        """
        return await self._get_async_loop().arun(user_input)

    def _get_async_loop(self):
        """
        Construit (et met en cache) un AsyncAgentLoop depuis la config existante.
        Évite de reconstruire à chaque appel arun().

        L'AsyncLLMClient est construit depuis le LLMClient sync via from_sync().
        """
        if hasattr(self, "_async_loop"):
            return self._async_loop

        from zeroagent.core.async_llm import AsyncLLMClient
        from zeroagent.harness.async_loop import AsyncAgentLoop

        async_llm = AsyncLLMClient.from_sync(self.llm)
        self._async_loop = AsyncAgentLoop(
            llm=async_llm,
            registry=self._registry,
            memory=self._memory,
            config=self._config,
            on_thought=self._loop._on_thought,
            on_action=self._loop._on_action,
            on_observation=self._loop._on_observation,
            on_final=self._loop._on_final,
            on_error=self._loop._on_error,
        )
        return self._async_loop

    def reset(self) -> None:
        """
        Vide la mémoire conversationnelle.
        Réinjecte le system prompt s'il était défini.
        """
        self._memory.clear()
        if self.system_prompt:
            self._memory.add(Message(role="system", content=self.system_prompt))

    def add_tool(self, fn: Callable) -> "Agent":
        """
        Ajoute un tool après la création de l'agent.
        Retourne self pour le chaining.
        """
        self._registry.register(fn)
        return self

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[str]:
        """Noms des tools enregistrés."""
        return [s["function"]["name"] for s in self._registry.get_schemas()]

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Agent("
            f"model={self.llm.model!r}, "
            f"tools={self.tools}, "
            f"memory={self._memory!r})"
        )
