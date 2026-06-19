"""
harness/parser.py — Parser les décisions du LLM.

Deux modes supportés :
  1. Tool calls structurés  : response.has_tool_calls == True
     -> le LLMClient a déjà parsé, on wrap juste dans ParsedAction
  2. ReAct texte libre      : le LLM génère Thought/Action/Action Input
     -> on extrait avec re (stdlib)

Le parser retourne toujours un ParsedAction, jamais d'exception.
En cas d'ambiguïté, on suppose une réponse finale.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from zeroagent.core.llm import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    TOOL_CALL    = "tool_call"      # le LLM veut appeler un tool
    FINAL_ANSWER = "final_answer"   # le LLM a sa réponse finale
    THOUGHT_ONLY = "thought_only"   # le LLM réfléchit sans agir (rare)


@dataclass
class ParsedAction:
    """
    Résultat normalisé du parsing d'une réponse LLM.

    Selon le type :
      TOOL_CALL    -> tool_name + arguments sont remplis
      FINAL_ANSWER -> final_answer est rempli
      THOUGHT_ONLY -> thought est rempli, rien d'autre
    """
    type: ActionType

    # Rempli si TOOL_CALL
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    tool_call_id: str = ""

    # Rempli si FINAL_ANSWER
    final_answer: str = ""

    # Toujours rempli si disponible
    thought: str = ""

    # Tool calls supplémentaires si le LLM en émet plusieurs d'un coup
    extra_tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_tool_call(self) -> bool:
        return self.type == ActionType.TOOL_CALL

    @property
    def is_final(self) -> bool:
        return self.type == ActionType.FINAL_ANSWER


# ---------------------------------------------------------------------------
# Patterns ReAct texte libre
# ---------------------------------------------------------------------------

_RE_THOUGHT      = re.compile(r"Thought\s*:\s*(.+?)(?=Action|Final Answer|$)", re.DOTALL | re.IGNORECASE)
_RE_ACTION       = re.compile(r"Action\s*:\s*(.+?)(?=Action Input|$)", re.DOTALL | re.IGNORECASE)
_RE_ACTION_INPUT = re.compile(r"Action Input\s*:\s*(.+?)(?=Observation|Thought|$)", re.DOTALL | re.IGNORECASE)
_RE_FINAL_ANSWER = re.compile(r"Final Answer\s*:\s*(.+)$", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict:
    """
    Extrait un dict JSON depuis une string.
    Accepte JSON pur, JSON dans backticks markdown, ou valeur scalaire.
    """
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return {"input": result}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {"input": text} if text else {}


# ---------------------------------------------------------------------------
# Parsing mode structuré
# ---------------------------------------------------------------------------

def _parse_structured(response: LLMResponse) -> ParsedAction:
    calls = response.tool_calls
    first = calls[0]
    return ParsedAction(
        type=ActionType.TOOL_CALL,
        tool_name=first.name,
        arguments=first.arguments,
        tool_call_id=first.id,
        thought=response.reasoning or "",
        extra_tool_calls=calls[1:] if len(calls) > 1 else [],
    )


# ---------------------------------------------------------------------------
# Parsing mode ReAct texte libre
# ---------------------------------------------------------------------------

def _parse_react_text(text: str) -> ParsedAction:
    thought = ""
    m = _RE_THOUGHT.search(text)
    if m:
        thought = m.group(1).strip()

    m = _RE_FINAL_ANSWER.search(text)
    if m:
        return ParsedAction(
            type=ActionType.FINAL_ANSWER,
            final_answer=m.group(1).strip(),
            thought=thought,
        )

    m_action = _RE_ACTION.search(text)
    m_input  = _RE_ACTION_INPUT.search(text)

    if m_action:
        tool_name = m_action.group(1).strip()
        arguments = _extract_json(m_input.group(1)) if m_input else {}
        return ParsedAction(
            type=ActionType.TOOL_CALL,
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
        )

    # Texte direct sans structure ReAct -> réponse finale
    if text.strip() and not thought:
        return ParsedAction(
            type=ActionType.FINAL_ANSWER,
            final_answer=text.strip(),
        )

    return ParsedAction(
        type=ActionType.THOUGHT_ONLY,
        thought=thought or text.strip(),
    )


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def parse_response(response: LLMResponse) -> ParsedAction:
    """
    Parse une LLMResponse -> ParsedAction normalisé.
    Détecte automatiquement le mode. Ne lève jamais d'exception.
    """
    if response.has_tool_calls:
        return _parse_structured(response)
    return _parse_react_text(response.content)
