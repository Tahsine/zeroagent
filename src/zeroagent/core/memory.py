"""
core/memory.py — Mémoire conversationnelle pure.

Deux implémentations sans dépendance LLM :
  - BufferMemory  : garde tous les messages
  - WindowMemory  : garde les N derniers messages (sliding window)

SummaryMemory (nécessite un LLM) vivra dans harness/memory.py.

Toutes les classes sont sérialisables en JSON pur via to_dict() / from_dict().

Usage:
    from zeroagent.core.memory import BufferMemory, WindowMemory
    from zeroagent.core.llm import Message

    mem = WindowMemory(k=10)
    mem.add(Message(role="user", content="Bonjour"))
    mem.add(Message(role="assistant", content="Bonjour !"))

    messages = mem.get()   # → list[Message] à injecter dans LLMClient
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from zeroagent.core.llm import Message


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

class BaseMemory(ABC):
    """
    Interface de base pour toute implémentation de mémoire.

    Contrat :
    - add()      : ajoute un message
    - get()      : retourne la liste de messages à envoyer au LLM
    - clear()    : remet la mémoire à zéro (garde le system prompt)
    - to_dict()  : sérialise en dict JSON-compatible
    - from_dict(): désérialise (classmethod)
    """

    @abstractmethod
    def add(self, message: Message) -> None:
        """Ajoute un message à la mémoire."""
        ...

    @abstractmethod
    def get(self) -> list[Message]:
        """
        Retourne la liste de messages à passer au LLM.
        L'ordre est toujours chronologique.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Vide la mémoire conversationnelle.
        Le system prompt (s'il existe) est conservé.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Sérialise la mémoire en dict JSON-compatible."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseMemory":
        """Désérialise depuis un dict."""
        ...

    def to_json(self) -> str:
        """Sérialise en JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "BaseMemory":
        """Désérialise depuis une JSON string."""
        return cls.from_dict(json.loads(s))

    def __len__(self) -> int:
        return len(self.get())

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict]:
        return [m.to_dict() for m in messages]

    @staticmethod
    def _dicts_to_messages(dicts: list[dict]) -> list[Message]:
        return [
            Message(
                role=d["role"],
                content=d["content"],
                tool_call_id=d.get("tool_call_id"),
                name=d.get("name"),
            )
            for d in dicts
        ]


# ---------------------------------------------------------------------------
# BufferMemory — garde tout
# ---------------------------------------------------------------------------

class BufferMemory(BaseMemory):
    """
    Mémoire tampon complète : conserve l'intégralité de la conversation.

    Adapté pour :
    - Conversations courtes
    - Contextes qui tiennent dans la fenêtre du LLM
    - Debug / développement

    Attention : sans limite, le contexte peut dépasser la fenêtre du LLM
    sur des conversations longues. Utiliser WindowMemory dans ce cas.

    Usage:
        mem = BufferMemory()
        mem.add(Message(role="system", content="Tu es un assistant."))
        mem.add(Message(role="user", content="Bonjour"))
        messages = mem.get()
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        """Vide tout sauf le premier message s'il est un system prompt."""
        if self._messages and self._messages[0].role == "system":
            system = self._messages[0]
            self._messages = [system]
        else:
            self._messages = []

    def to_dict(self) -> dict:
        return {
            "type": "BufferMemory",
            "messages": self._messages_to_dicts(self._messages),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BufferMemory":
        mem = cls()
        mem._messages = cls._dicts_to_messages(data.get("messages", []))
        return mem

    def __repr__(self) -> str:
        return f"BufferMemory(messages={len(self._messages)})"


# ---------------------------------------------------------------------------
# WindowMemory — sliding window des N derniers messages
# ---------------------------------------------------------------------------

class WindowMemory(BaseMemory):
    """
    Mémoire à fenêtre glissante : garde les `k` derniers messages.

    Le system prompt (premier message de role="system") est TOUJOURS
    conservé, même s'il sort de la fenêtre. Il est réinjecté en tête
    à chaque appel à get().

    Adapté pour :
    - Conversations longues où le contexte complet ne rentre pas
    - Limiter les coûts en tokens
    - La majorité des cas d'usage en production

    Usage:
        mem = WindowMemory(k=10)
        mem.add(Message(role="system", content="Tu es un assistant."))
        for msg in conversation:
            mem.add(msg)
        # get() retourne : [system] + 10 derniers messages non-system
    """

    def __init__(self, k: int = 20) -> None:
        if k < 1:
            raise ValueError(f"k doit être >= 1, reçu : {k}")
        self.k = k
        self._system: Message | None = None
        self._messages: list[Message] = []   # ne contient jamais le system

    def add(self, message: Message) -> None:
        """
        Ajoute un message.
        Si c'est le premier message system, il est stocké séparément.
        Si un system est ajouté plus tard, il remplace le précédent.
        """
        if message.role == "system":
            self._system = message
            return
        self._messages.append(message)

    def get(self) -> list[Message]:
        """
        Retourne [system (si existe)] + les k derniers messages non-system.
        """
        window = self._messages[-self.k:]
        if self._system is not None:
            return [self._system] + window
        return list(window)

    def clear(self) -> None:
        """Vide les messages de la fenêtre. Garde le system prompt."""
        self._messages = []

    @property
    def total_stored(self) -> int:
        """Nombre total de messages stockés (hors system)."""
        return len(self._messages)

    @property
    def is_full(self) -> bool:
        """True si la fenêtre a atteint sa capacité maximale."""
        return len(self._messages) >= self.k

    def to_dict(self) -> dict:
        return {
            "type": "WindowMemory",
            "k": self.k,
            "system": self._system.to_dict() if self._system else None,
            "messages": self._messages_to_dicts(self._messages),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WindowMemory":
        mem = cls(k=data.get("k", 20))
        system_dict = data.get("system")
        if system_dict:
            mem._system = Message(
                role=system_dict["role"],
                content=system_dict["content"],
            )
        mem._messages = cls._dicts_to_messages(data.get("messages", []))
        return mem

    def __repr__(self) -> str:
        return (
            f"WindowMemory(k={self.k}, "
            f"stored={self.total_stored}, "
            f"has_system={self._system is not None})"
        )
