"""
Point d'entrée public de zeroagent.

Imports publics stables — utilisables directement depuis le package racine :

    from zeroagent import Agent, tool, LLMClient
    from zeroagent import BufferMemory, WindowMemory, SummaryMemory
    from zeroagent import Message, LLMResponse, ToolCall, ToolRegistry

Imports avancés (toujours disponibles via leur module) :
    from zeroagent.core.llm import LLMClient
    from zeroagent.harness.loop import AgentLoop, RunConfig, RunResult, StopReason
    from zeroagent.harness.parser import parse_response, ParsedAction, ActionType
    from zeroagent.harness.executor import Executor, ExecutionResult
"""

# ---------------------------------------------------------------------------
# Core — LLM client + types
# ---------------------------------------------------------------------------
from zeroagent.core.llm import (
    LLMClient,
    Message,
    LLMResponse,
    ToolCall,
)

# ---------------------------------------------------------------------------
# Core — tools
# ---------------------------------------------------------------------------
from zeroagent.core.tools import (
    tool,
    ToolRegistry,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Core — memory
# ---------------------------------------------------------------------------
from zeroagent.core.memory import (
    BaseMemory,
    BufferMemory,
    WindowMemory,
)

# ---------------------------------------------------------------------------
# Harness — memory étendue (SummaryMemory nécessite un LLM optionnel)
# ---------------------------------------------------------------------------
from zeroagent.harness.memory import SummaryMemory

# ---------------------------------------------------------------------------
# Harness — Agent (façade principale)
# ---------------------------------------------------------------------------
from zeroagent.harness.agent import Agent

# ---------------------------------------------------------------------------
# Harness — types utiles pour les hooks et introspection
# ---------------------------------------------------------------------------
from zeroagent.harness.loop import RunResult, StopReason, RunConfig
from zeroagent.harness.parser import ParsedAction, ActionType
from zeroagent.harness.executor import ExecutionResult

# ---------------------------------------------------------------------------
# Async — LLM client + boucle agent
# ---------------------------------------------------------------------------
from zeroagent.core.async_llm import AsyncLLMClient
from zeroagent.harness.async_loop import AsyncAgentLoop, AsyncExecutor

# ---------------------------------------------------------------------------
# Surface publique explicite
# ---------------------------------------------------------------------------
__all__ = [
    # Entrées principales — ce que 90% des utilisateurs importent
    "Agent",
    "tool",
    "LLMClient",

    # Types LLM
    "Message",
    "LLMResponse",
    "ToolCall",

    # Tools
    "ToolRegistry",
    "ToolSchema",

    # Mémoire
    "BaseMemory",
    "BufferMemory",
    "WindowMemory",
    "SummaryMemory",

    # Résultats et contrôle de boucle
    "RunResult",
    "StopReason",
    "RunConfig",

    # Parsing et exécution (pour utilisateurs avancés / hooks)
    "ParsedAction",
    "ActionType",
    "ExecutionResult",

    # Async
    "AsyncLLMClient",
    "AsyncAgentLoop",
    "AsyncExecutor",
]
