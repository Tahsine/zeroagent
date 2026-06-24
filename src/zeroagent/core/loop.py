"""
harness/loop.py — Boucle ReAct.

Orchestre LLMClient + Parser + Executor + Memory dans la boucle :

  1. memory.get() → messages
  2. llm.complete(messages, tools) → LLMResponse
  3. parser.parse_response(response) → ParsedAction
  4. si final_answer → STOP
  5. executor.execute(action) → list[ExecutionResult]
  6. memory.add(messages assistant + tool results)
  7. → retour à 1

Guards :
  - max_iterations : évite les boucles infinies
  - empty_response : si le LLM répond vide plusieurs fois
  - thought_only_limit : si le LLM réfléchit sans jamais agir

Hooks (tous optionnels) :
  - on_start(user_input)
  - on_thought(thought, iteration)
  - on_action(tool_name, arguments, iteration)
  - on_observation(tool_name, observation, success, iteration)
  - on_final(answer, stop_reason)
  - on_error(error, iteration)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from zeroagent.core.llm import LLMClient, Message
from zeroagent.core.memory import BaseMemory, BufferMemory
from zeroagent.core.tools import ToolRegistry
from zeroagent.harness.executor import Executor, ExecutionResult
from zeroagent.harness.parser import ActionType, ParsedAction, parse_response


# ---------------------------------------------------------------------------
# Config et raison d'arrêt
# ---------------------------------------------------------------------------

class StopReason(str, Enum):
    FINAL_ANSWER   = "final_answer"     # le LLM a donné sa réponse
    MAX_ITERATIONS = "max_iterations"   # seuil atteint
    EMPTY_RESPONSE = "empty_response"   # LLM répond vide trop souvent
    ERROR          = "error"            # exception non récupérable


@dataclass
class RunConfig:
    """Configuration d'un run agent."""
    max_iterations: int    = 10
    verbose: bool          = False
    # Nombre max d'itérations "thought only" consécutives avant stop
    max_thought_only: int  = 3
    # Nombre max de réponses vides consécutives avant stop
    max_empty: int         = 2


@dataclass
class RunResult:
    """Résultat complet d'un run agent."""
    answer: str
    stop_reason: StopReason
    iterations: int
    thoughts: list[str]       = field(default_factory=list)
    tool_calls_made: int      = 0
    success: bool             = True

    def __repr__(self) -> str:
        return (
            f"RunResult("
            f"stop={self.stop_reason.value}, "
            f"iterations={self.iterations}, "
            f"tools_called={self.tool_calls_made})"
        )


# ---------------------------------------------------------------------------
# Types des hooks
# ---------------------------------------------------------------------------

OnStartFn       = Callable[[str], None]
OnThoughtFn     = Callable[[str, int], None]
OnActionFn      = Callable[[str, dict, int], None]
OnObservationFn = Callable[[str, str, bool, int], None]
OnFinalFn       = Callable[[str, StopReason], None]
OnErrorFn       = Callable[[Exception, int], None]


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Boucle ReAct complète.

    Ne pas utiliser directement — passer par Agent (harness/agent.py)
    qui assemble correctement tous les composants.

    Usage interne :
        loop = AgentLoop(
            llm=llm_client,
            registry=tool_registry,
            memory=memory,
            config=RunConfig(max_iterations=10, verbose=True),
        )
        result = loop.run("Quelle est la capitale du Bénin ?")
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        memory: BaseMemory,
        config: RunConfig | None = None,
        # Hooks optionnels
        on_start: OnStartFn | None = None,
        on_thought: OnThoughtFn | None = None,
        on_action: OnActionFn | None = None,
        on_observation: OnObservationFn | None = None,
        on_final: OnFinalFn | None = None,
        on_error: OnErrorFn | None = None,
    ) -> None:
        self.llm      = llm
        self.registry = registry
        self.memory   = memory
        self.config   = config or RunConfig()
        self.executor = Executor(registry)

        # Hooks
        self._on_start       = on_start
        self._on_thought     = on_thought
        self._on_action      = on_action
        self._on_observation = on_observation
        self._on_final       = on_final
        self._on_error       = on_error

    # ------------------------------------------------------------------
    # Run principal
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> RunResult:
        """
        Lance la boucle ReAct pour un user_input.
        Retourne un RunResult avec la réponse finale et les métadonnées.
        """
        self._fire(self._on_start, user_input)

        # Injecter le message utilisateur en mémoire
        self.memory.add(Message(role="user", content=user_input))

        thoughts: list[str] = []
        tool_calls_made = 0
        thought_only_streak = 0
        empty_streak = 0

        tools_schemas = self.registry.get_schemas() or None

        for iteration in range(1, self.config.max_iterations + 1):

            # ----------------------------------------------------------
            # 1. Appel LLM
            # ----------------------------------------------------------
            try:
                messages  = self.memory.get()
                response  = self.llm.complete(messages, tools=tools_schemas)
            except Exception as e:
                self._fire(self._on_error, e, iteration)
                if self.config.verbose:
                    print(f"[loop] Erreur LLM itération {iteration}: {e}")
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
                    return self._stop(
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
            self._add_assistant_message(response)

            # ----------------------------------------------------------
            # 3. Parser la décision
            # ----------------------------------------------------------
            action = parse_response(response)

            # Logs verbose
            if self.config.verbose:
                self._print_verbose(action, iteration)

            # ----------------------------------------------------------
            # 4. Thought
            # ----------------------------------------------------------
            if action.thought:
                thoughts.append(action.thought)
                self._fire(self._on_thought, action.thought, iteration)

            # ----------------------------------------------------------
            # 5. Final Answer → STOP
            # ----------------------------------------------------------
            if action.is_final:
                return self._stop(
                    answer=action.final_answer,
                    reason=StopReason.FINAL_ANSWER,
                    iteration=iteration,
                    thoughts=thoughts,
                    tool_calls_made=tool_calls_made,
                )

            # ----------------------------------------------------------
            # 6. Thought only (pas d'action) → streak guard
            # ----------------------------------------------------------
            if action.type == ActionType.THOUGHT_ONLY:
                thought_only_streak += 1
                if thought_only_streak >= self.config.max_thought_only:
                    return self._stop(
                        answer=action.thought,
                        reason=StopReason.MAX_ITERATIONS,
                        iteration=iteration,
                        thoughts=thoughts,
                        tool_calls_made=tool_calls_made,
                    )
                continue
            thought_only_streak = 0

            # ----------------------------------------------------------
            # 7. Tool call → exécuter + injecter observations
            # ----------------------------------------------------------
            results = self.executor.execute(action)
            tool_calls_made += len(results)

            for result in results:
                self._fire(
                    self._on_action,
                    result.tool_call.name,
                    result.tool_call.arguments,
                    iteration,
                )
                self._fire(
                    self._on_observation,
                    result.tool_call.name,
                    result.observation,
                    result.success,
                    iteration,
                )
                self._add_tool_result(result)

        # ------------------------------------------------------------------
        # Max iterations atteint
        # ------------------------------------------------------------------
        return self._stop(
            answer="Nombre maximum d'itérations atteint sans réponse finale.",
            reason=StopReason.MAX_ITERATIONS,
            iteration=self.config.max_iterations,
            thoughts=thoughts,
            tool_calls_made=tool_calls_made,
        )

    # ------------------------------------------------------------------
    # Gestion des messages mémoire
    # ------------------------------------------------------------------

    def _add_assistant_message(self, response) -> None:
        """
        Ajoute le message assistant à la mémoire.

        Pour OpenAI/Anthropic/Ollama avec tool calls, le message assistant
        DOIT contenir les tool_calls pour que l'API accepte les messages
        tool suivants — sans ça, l'observation n'est pas reliée à la
        décision du LLM, et certains providers (ex: minimax via Ollama)
        ignorent l'observation et rappellent le même tool en boucle.

        Le contenu peut être vide si le LLM ne génère que des tool calls.
        """
        content = response.content or ""
        self.memory.add(Message(
            role="assistant",
            content=content,
            tool_calls=list(response.tool_calls),
        ))

    def _add_tool_result(self, result: ExecutionResult) -> None:
        """
        Injecte le résultat d'un tool comme message en mémoire.

        Format : role="tool" avec tool_call_id pour OpenAI/Anthropic.
        Si pas d'id (mode ReAct texte libre) → role="user" avec
        un format "Observation: ..." pour que le LLM comprenne.
        """
        tc = result.tool_call

        if tc.id and tc.id != "react-0":
            # Mode structuré : message tool officiel
            self.memory.add(Message(
                role="tool",
                content=result.observation,
                tool_call_id=tc.id,
                name=tc.name,
            ))
        else:
            # Mode ReAct texte libre : injecter comme user message
            self.memory.add(Message(
                role="user",
                content=f"Observation: {result.observation}",
            ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stop(
        self,
        answer: str,
        reason: StopReason,
        iteration: int,
        thoughts: list[str],
        tool_calls_made: int,
    ) -> RunResult:
        self._fire(self._on_final, answer, reason)
        return RunResult(
            answer=answer,
            stop_reason=reason,
            iterations=iteration,
            thoughts=thoughts,
            tool_calls_made=tool_calls_made,
            success=(reason == StopReason.FINAL_ANSWER),
        )

    def _fire(self, hook, *args) -> None:
        """Appelle un hook si défini, ignore les erreurs."""
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:
            pass

    def _print_verbose(self, action: ParsedAction, iteration: int) -> None:
        print(f"\n[loop] Itération {iteration}")
        if action.thought:
            print(f"  Thought: {action.thought[:120]}")
        if action.type == ActionType.TOOL_CALL:
            print(f"  Action: {action.tool_name}({action.arguments})")
        elif action.type == ActionType.FINAL_ANSWER:
            print(f"  Final Answer: {action.final_answer[:120]}")

    def __repr__(self) -> str:
        return (
            f"AgentLoop("
            f"model={self.llm.model!r}, "
            f"tools={len(self.registry)}, "
            f"max_iter={self.config.max_iterations})"
        )
