"""
harness/async_loop.py — Boucle ReAct async native.

Même logique que AgentLoop (loop.py) mais entièrement non-bloquante.
Utilise AsyncLLMClient pour les appels LLM et asyncio.gather()
pour l'exécution parallèle des tool calls dans une même itération.

Architecture :
  - AsyncAgentLoop.arun()      → coroutine, retourne RunResult
  - AsyncExecutor.aexecute()   → exécute les tools en parallèle via gather()

Pourquoi la parallélisation des tools est importante :
  Quand le LLM retourne plusieurs tool calls en une seule réponse
  (ex: "cherche X ET cherche Y en même temps"), l'exécution séquentielle
  les force à s'enchaîner. Avec gather(), ils tournent en concurrence —
  gain réel si les tools font de l'I/O (HTTP, fichier, DB).

Hooks async :
  Les hooks peuvent être des coroutines ou des fonctions sync.
  AsyncAgentLoop les détecte et les appelle correctement dans les deux cas.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Callable

from zeroagent.core.llm import Message, ToolCall
from zeroagent.core.memory import BaseMemory
from zeroagent.core.tools import ToolRegistry
from zeroagent.harness.executor import ExecutionResult
from zeroagent.harness.loop import RunConfig, RunResult, StopReason
from zeroagent.harness.parser import ActionType, ParsedAction, parse_response


# ---------------------------------------------------------------------------
# AsyncExecutor — exécution parallèle des tool calls
# ---------------------------------------------------------------------------

class AsyncExecutor:
    """
    Exécute les tool calls d'une ParsedAction en parallèle via asyncio.gather().

    Pour les tools sync (@tool normal), les fonctions sont wrappées dans
    run_in_executor() pour ne pas bloquer l'event loop.

    Pour les tools async (coroutines), ils sont awaités directement.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def aexecute(self, action: ParsedAction) -> list[ExecutionResult]:
        """
        Exécute tous les tool calls de l'action en parallèle.
        Le tool call principal est dans action.tool_name/arguments/tool_call_id.
        Les tool calls supplémentaires sont dans action.extra_tool_calls.
        Retourne une liste de ExecutionResult dans le même ordre.
        """
        if action.type != ActionType.TOOL_CALL:
            return []

        # Construire la liste complète des tool calls
        primary = ToolCall(
            name=action.tool_name,
            arguments=action.arguments,
            id=action.tool_call_id or "tc-0",
        )
        all_calls = [primary] + list(action.extra_tool_calls)

        tasks = [self._run_one(tc) for tc in all_calls]
        return list(await asyncio.gather(*tasks))

    async def _run_one(self, tool_call) -> ExecutionResult:
        """Exécute un seul tool call, sync ou async."""
        tool_name = tool_call.name
        args = tool_call.arguments or {}

        # Récupérer le ToolSchema (None si tool inconnu)
        tool_schema = self.registry._tools.get(tool_name)
        if tool_schema is None:
            return ExecutionResult(
                tool_call=tool_call,
                observation=f"Outil inconnu : '{tool_name}'",
                success=False,
            )

        fn = tool_schema.fn  # fonction originale, pas le wrapper

        try:
            if inspect.iscoroutinefunction(fn):
                # Tool async natif — awaité directement
                result = await fn(**args)
            else:
                # Tool sync — exécuté dans un thread pour ne pas bloquer l'event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: fn(**args))

            return ExecutionResult(
                tool_call=tool_call,
                observation=str(result),
                success=True,
            )
        except TypeError as e:
            return ExecutionResult(
                tool_call=tool_call,
                observation=f"Arguments invalides pour '{tool_name}': {e}",
                success=False,
            )
        except Exception as e:
            return ExecutionResult(
                tool_call=tool_call,
                observation=f"Erreur dans '{tool_name}': {e}",
                success=False,
            )


# ---------------------------------------------------------------------------
# AsyncAgentLoop
# ---------------------------------------------------------------------------

class AsyncAgentLoop:
    """
    Boucle ReAct async native.

    Utilise AsyncLLMClient pour les appels LLM et AsyncExecutor
    pour l'exécution parallèle des tool calls.

    Ne pas utiliser directement — passer par Agent.arun() qui
    assemble correctement tous les composants.

    Usage interne :
        loop = AsyncAgentLoop(
            llm=async_llm_client,
            registry=tool_registry,
            memory=memory,
            config=RunConfig(max_iterations=10),
        )
        result = await loop.arun("Quelle est la capitale du Bénin ?")
    """

    def __init__(
        self,
        llm,                           # AsyncLLMClient
        registry: ToolRegistry,
        memory: BaseMemory,
        config: RunConfig | None = None,
        on_start: Callable | None = None,
        on_thought: Callable | None = None,
        on_action: Callable | None = None,
        on_observation: Callable | None = None,
        on_final: Callable | None = None,
        on_error: Callable | None = None,
    ) -> None:
        self.llm      = llm
        self.registry = registry
        self.memory   = memory
        self.config   = config or RunConfig()
        self.executor = AsyncExecutor(registry)

        self._on_start       = on_start
        self._on_thought     = on_thought
        self._on_action      = on_action
        self._on_observation = on_observation
        self._on_final       = on_final
        self._on_error       = on_error

    # ------------------------------------------------------------------
    # Run principal
    # ------------------------------------------------------------------

    async def arun(self, user_input: str) -> RunResult:
        """
        Lance la boucle ReAct async pour un user_input.
        Retourne un RunResult avec la réponse finale et les métadonnées.
        """
        await self._fire(self._on_start, user_input)

        self.memory.add(Message(role="user", content=user_input))

        thoughts: list[str] = []
        tool_calls_made = 0
        thought_only_streak = 0
        empty_streak = 0

        tools_schemas = self.registry.get_schemas() or None

        for iteration in range(1, self.config.max_iterations + 1):

            # ----------------------------------------------------------
            # 1. Appel LLM async
            # ----------------------------------------------------------
            try:
                messages = self.memory.get()
                response = await self.llm.acomplete(messages, tools=tools_schemas)
            except Exception as e:
                await self._fire(self._on_error, e, iteration)
                if self.config.verbose:
                    print(f"[async_loop] Erreur LLM itération {iteration}: {e}")
                return RunResult(
                    answer=f"Erreur LLM: {e}",
                    stop_reason=StopReason.ERROR,
                    iterations=iteration,
                    thoughts=thoughts,
                    tool_calls_made=tool_calls_made,
                    success=False,
                )

            # Réponse vide
            if not response.content and not response.has_tool_calls:
                empty_streak += 1
                if empty_streak >= self.config.max_empty:
                    return self._make_stop(
                        answer="",
                        reason=StopReason.EMPTY_RESPONSE,
                        iteration=iteration,
                        thoughts=thoughts,
                        tool_calls_made=tool_calls_made,
                    )
                continue
            empty_streak = 0

            # ----------------------------------------------------------
            # 2. Ajouter la réponse assistant en mémoire
            # ----------------------------------------------------------
            # Si la réponse contient des tool calls structurés, on doit
            # injecter le message assistant AVEC les tool_calls — sinon
            # les messages role='tool' suivants sont orphelins et le LLM
            # ne voit jamais les observations.
            self.memory.add(Message(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls if response.has_tool_calls else [],
            ))

            # ----------------------------------------------------------
            # 3. Parser la décision
            # ----------------------------------------------------------
            action = parse_response(response)

            if self.config.verbose:
                self._print_verbose(action, iteration)

            # ----------------------------------------------------------
            # 4. Thought
            # ----------------------------------------------------------
            if action.thought:
                thoughts.append(action.thought)
                await self._fire(self._on_thought, action.thought, iteration)

            # ----------------------------------------------------------
            # 5. Final Answer → STOP
            # ----------------------------------------------------------
            if action.is_final:
                await self._fire(self._on_final, action.final_answer, StopReason.FINAL_ANSWER)
                return self._make_stop(
                    answer=action.final_answer,
                    reason=StopReason.FINAL_ANSWER,
                    iteration=iteration,
                    thoughts=thoughts,
                    tool_calls_made=tool_calls_made,
                )

            # ----------------------------------------------------------
            # 6. Thought only → streak guard
            # ----------------------------------------------------------
            if action.type == ActionType.THOUGHT_ONLY:
                thought_only_streak += 1
                if thought_only_streak >= self.config.max_thought_only:
                    return self._make_stop(
                        answer=action.thought or "",
                        reason=StopReason.MAX_ITERATIONS,
                        iteration=iteration,
                        thoughts=thoughts,
                        tool_calls_made=tool_calls_made,
                    )
                continue
            thought_only_streak = 0

            # ----------------------------------------------------------
            # 7. Tool calls → exécution parallèle via asyncio.gather()
            # ----------------------------------------------------------
            results = await self.executor.aexecute(action)
            tool_calls_made += len(results)

            for result in results:
                await self._fire(
                    self._on_action,
                    result.tool_call.name,
                    result.tool_call.arguments,
                    iteration,
                )
                await self._fire(
                    self._on_observation,
                    result.tool_call.name,
                    result.observation,
                    result.success,
                    iteration,
                )
                self._add_tool_result(result)

        # Max iterations
        await self._fire(
            self._on_final,
            "Nombre maximum d'itérations atteint sans réponse finale.",
            StopReason.MAX_ITERATIONS,
        )
        return self._make_stop(
            answer="Nombre maximum d'itérations atteint sans réponse finale.",
            reason=StopReason.MAX_ITERATIONS,
            iteration=self.config.max_iterations,
            thoughts=thoughts,
            tool_calls_made=tool_calls_made,
        )

    # ------------------------------------------------------------------
    # Gestion mémoire (identique à AgentLoop)
    # ------------------------------------------------------------------

    def _add_tool_result(self, result: ExecutionResult) -> None:
        tc = result.tool_call
        if tc.id and tc.id != "react-0":
            self.memory.add(Message(
                role="tool",
                content=result.observation,
                tool_call_id=tc.id,
                name=tc.name,
            ))
        else:
            self.memory.add(Message(
                role="user",
                content=f"Observation: {result.observation}",
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_stop(
        self,
        answer: str,
        reason: StopReason,
        iteration: int,
        thoughts: list[str],
        tool_calls_made: int,
    ) -> RunResult:
        return RunResult(
            answer=answer,
            stop_reason=reason,
            iterations=iteration,
            thoughts=thoughts,
            tool_calls_made=tool_calls_made,
            success=(reason == StopReason.FINAL_ANSWER),
        )

    async def _fire(self, hook: Callable | None, *args) -> None:
        """
        Appelle un hook — supporte sync et async.
        Ignore les erreurs pour ne pas crasher la boucle.
        """
        if hook is None:
            return
        try:
            result = hook(*args)
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    def _print_verbose(self, action: ParsedAction, iteration: int) -> None:
        print(f"\n[async_loop] Itération {iteration}")
        if action.thought:
            print(f"  Thought: {action.thought[:120]}")
        if action.type == ActionType.TOOL_CALL:
            print(f"  Action: {action.tool_name}({action.arguments})")
        elif action.type == ActionType.FINAL_ANSWER:
            print(f"  Final Answer: {action.final_answer[:120]}")

    def __repr__(self) -> str:
        return (
            f"AsyncAgentLoop("
            f"model={self.llm.model!r}, "
            f"tools={len(self.registry)}, "
            f"max_iter={self.config.max_iterations})"
        )
