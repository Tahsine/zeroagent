"""
harness/executor.py — Exécute les actions du harness.

Prend un ParsedAction, appelle le bon tool via ToolRegistry,
retourne toujours une observation str.

Principe : ne jamais crasher le loop.
Toute erreur d'exécution devient une observation d'erreur lisible
que le LLM peut voir et corriger à l'itération suivante.
"""

from __future__ import annotations

from dataclasses import dataclass

from zeroagent.core.llm import ToolCall
from zeroagent.core.tools import ToolRegistry
from zeroagent.harness.parser import ActionType, ParsedAction


# ---------------------------------------------------------------------------
# Résultat d'exécution
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Résultat de l'exécution d'un tool.

    Toujours produit, même en cas d'erreur.
    Le loop l'injecte dans la mémoire comme message "tool" ou "user".
    """
    tool_call: ToolCall       # le tool call qui a été exécuté
    observation: str          # résultat (ou message d'erreur)
    success: bool             # False si le tool a levé une exception

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ExecutionResult({status} {self.tool_call.name} → {self.observation[:60]!r})"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Exécute les tool calls depuis un ParsedAction.

    Usage:
        executor = Executor(registry)
        results = executor.execute(parsed_action)
        for result in results:
            print(result.tool_call.name, "→", result.observation)
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, action: ParsedAction) -> list[ExecutionResult]:
        """
        Exécute le tool call d'un ParsedAction.

        Retourne une liste de ExecutionResult (toujours 1 élément
        pour l'instant — le parallel tool call arrivera en Phase 4).

        Ne lève jamais d'exception : les erreurs deviennent des observations.
        """
        if action.type != ActionType.TOOL_CALL:
            return []

        # Reconstruire un ToolCall depuis les champs de ParsedAction
        tc = ToolCall(
            id=action.tool_call_id or f"call_{action.tool_name}",
            name=action.tool_name,
            arguments=action.arguments,
        )
        return [self._execute_one(tc)]

    def _execute_one(self, tc: ToolCall) -> ExecutionResult:
        """
        Exécute un seul ToolCall. Capture toutes les exceptions.
        """
        # Tool inconnu
        if tc.name not in self.registry:
            observation = (
                f"Erreur : tool '{tc.name}' introuvable. "
                f"Tools disponibles : {self._available()}."
            )
            return ExecutionResult(tool_call=tc, observation=observation, success=False)

        # Exécution
        try:
            observation = self.registry.execute(tc.name, tc.arguments)
            return ExecutionResult(tool_call=tc, observation=observation, success=True)
        except RuntimeError as e:
            observation = f"Erreur lors de l'exécution de '{tc.name}': {e}"
            return ExecutionResult(tool_call=tc, observation=observation, success=False)
        except Exception as e:
            observation = f"Erreur inattendue dans '{tc.name}': {type(e).__name__}: {e}"
            return ExecutionResult(tool_call=tc, observation=observation, success=False)

    def _available(self) -> str:
        schemas = self.registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        return ", ".join(names) if names else "aucun"

    def __repr__(self) -> str:
        return f"Executor(registry={self.registry!r})"
