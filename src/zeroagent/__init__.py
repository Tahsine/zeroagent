"""
zeroagent — Agent IA + Workflow SDK. Zero dependencies.

Usage:
    from zeroagent import Agent, tool, LLMClient
    from zeroagent import BufferMemory, WindowMemory, SummaryMemory
"""

__version__ = "0.1.0"

from zeroagent.sdk import (
    # Entrées principales
    Agent,
    tool,
    LLMClient,

    # Types LLM
    Message,
    LLMResponse,
    ToolCall,

    # Tools
    ToolRegistry,
    ToolSchema,

    # Mémoire
    BaseMemory,
    BufferMemory,
    WindowMemory,
    SummaryMemory,

    # Résultats et contrôle
    RunResult,
    StopReason,
    RunConfig,

    # Avancé
    ParsedAction,
    ActionType,
    ExecutionResult,

    # Async
    AsyncLLMClient,
    AsyncAgentLoop,
    AsyncExecutor,

    # Workflow
    State,
    Node,
    Graph,
    GraphResult,
    Edge,
    ConditionalEdge,
)

__all__ = [
    "__version__",
    "Agent",
    "tool",
    "LLMClient",
    "Message",
    "LLMResponse",
    "ToolCall",
    "ToolRegistry",
    "ToolSchema",
    "BaseMemory",
    "BufferMemory",
    "WindowMemory",
    "SummaryMemory",
    "RunResult",
    "StopReason",
    "RunConfig",
    "ParsedAction",
    "ActionType",
    "ExecutionResult",
    "AsyncLLMClient",
    "AsyncAgentLoop",
    "AsyncExecutor",
    "State",
    "Node",
    "Graph",
    "GraphResult",
    "Edge",
    "ConditionalEdge",
]