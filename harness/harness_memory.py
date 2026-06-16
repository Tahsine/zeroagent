"""
harness/memory.py — SummaryMemory.

Mémoire conversationnelle avec résumé automatique via LLM.
Vit dans harness/ (pas core/) parce qu'elle dépend de LLMClient.

Quand le nombre de messages dépasse max_messages :
  - Les messages les plus anciens (hors system) sont envoyés au LLM
    pour résumé.
  - Le résumé remplace ces anciens messages par un seul message
    role="system" préfixé "Résumé :".
  - Les messages récents sont conservés intacts.

Si llm=None → trim brutal (comportement WindowMemory).
"""

from __future__ import annotations

import json

from zeroagent.core.llm import LLMClient, Message
from zeroagent.core.memory import BaseMemory


# Prompt de résumé par défaut
_DEFAULT_SUMMARY_PROMPT = (
    "Tu es un assistant chargé de résumer des conversations. "
    "Voici une série de messages entre un utilisateur et un assistant. "
    "Produis un résumé concis (5-10 phrases max) qui capture les points clés, "
    "les décisions prises et les informations importantes. "
    "Réponds uniquement avec le résumé, sans préambule."
)


class SummaryMemory(BaseMemory):
    """
    Mémoire avec résumé LLM automatique des anciens messages.

    Paramètres :
        llm            : LLMClient pour générer les résumés.
                         Si None, trim brutal (comme WindowMemory).
        max_messages   : seuil avant déclenchement du résumé.
        keep_recent    : nombre de messages récents à conserver intacts
                         après résumé (default: max_messages // 2).
        summary_prompt : system prompt pour le LLM de résumé.
        summary_model  : si fourni, overrides llm.model pour les résumés
                         (utile pour utiliser un modèle moins cher).

    Usage:
        llm = LLMClient(base_url="...", model="gpt-4o-mini", api_key="...")
        mem = SummaryMemory(llm=llm, max_messages=20)

        mem.add(Message(role="system", content="Tu es un assistant."))
        # ... beaucoup de messages ...
        messages = mem.get()   # résumé déclenché automatiquement si besoin
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        max_messages: int = 20,
        keep_recent: int | None = None,
        summary_prompt: str = _DEFAULT_SUMMARY_PROMPT,
        summary_model: str | None = None,
    ) -> None:
        if max_messages < 4:
            raise ValueError("max_messages doit être >= 4")

        self.llm = llm
        self.max_messages = max_messages
        self.keep_recent = keep_recent or max(max_messages // 2, 2)
        self.summary_prompt = summary_prompt
        self.summary_model = summary_model

        self._system: Message | None = None
        self._messages: list[Message] = []   # sans le system
        self._summary: str | None = None     # dernier résumé généré

    # ------------------------------------------------------------------
    # Interface BaseMemory
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        if message.role == "system":
            self._system = message
            return
        self._messages.append(message)
        self._maybe_summarize()

    def get(self) -> list[Message]:
        """
        Retourne [system?] + [résumé?] + messages récents.
        Le résumé est injecté comme message assistant pour ne pas
        confondre le LLM avec deux system prompts.
        """
        result: list[Message] = []

        if self._system:
            result.append(self._system)

        if self._summary:
            result.append(Message(
                role="assistant",
                content=f"[Résumé de la conversation précédente]\n{self._summary}",
            ))

        result.extend(self._messages[-self.keep_recent:])
        return result

    def clear(self) -> None:
        """Vide les messages. Garde system et résumé."""
        self._messages = []

    def reset_summary(self) -> None:
        """Vide aussi le résumé accumulé."""
        self._summary = None
        self._messages = []

    # ------------------------------------------------------------------
    # Résumé automatique
    # ------------------------------------------------------------------

    def _maybe_summarize(self) -> None:
        """Déclenche le résumé si le seuil est dépassé."""
        if len(self._messages) <= self.max_messages:
            return

        # Séparer les anciens messages à résumer des récents à garder
        n_to_summarize = len(self._messages) - self.keep_recent
        to_summarize = self._messages[:n_to_summarize]
        self._messages = self._messages[n_to_summarize:]

        if self.llm is not None:
            new_summary = self._summarize_with_llm(to_summarize)
        else:
            # Trim brutal sans LLM
            new_summary = self._summarize_naive(to_summarize)

        # Concaténer avec le résumé précédent s'il existe
        if self._summary:
            self._summary = f"{self._summary}\n\n{new_summary}"
        else:
            self._summary = new_summary

    def _summarize_with_llm(self, messages: list[Message]) -> str:
        """Envoie les messages au LLM pour résumé."""
        # Formater les messages comme texte lisible
        conversation = "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in messages
            if m.content
        )

        # Utiliser le modèle de résumé si spécifié
        original_model = self.llm.model
        if self.summary_model:
            self.llm.model = self.summary_model

        try:
            response = self.llm.complete([
                Message(role="system", content=self.summary_prompt),
                Message(role="user", content=conversation),
            ])
            return response.content.strip()
        except Exception as e:
            # En cas d'erreur LLM → fallback naïf
            return self._summarize_naive(messages)
        finally:
            self.llm.model = original_model

    def _summarize_naive(self, messages: list[Message]) -> str:
        """
        Résumé sans LLM : concatène les contenus tronqués.
        Utilisé si llm=None ou si le LLM échoue.
        """
        parts = []
        for m in messages:
            if m.content:
                content = m.content[:100] + "..." if len(m.content) > 100 else m.content
                parts.append(f"{m.role}: {content}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "SummaryMemory",
            "max_messages": self.max_messages,
            "keep_recent": self.keep_recent,
            "summary_prompt": self.summary_prompt,
            "system": self._system.to_dict() if self._system else None,
            "summary": self._summary,
            "messages": self._messages_to_dicts(self._messages),
        }

    @classmethod
    def from_dict(cls, data: dict, llm: LLMClient | None = None) -> "SummaryMemory":
        mem = cls(
            llm=llm,
            max_messages=data.get("max_messages", 20),
            keep_recent=data.get("keep_recent", None),
            summary_prompt=data.get("summary_prompt", _DEFAULT_SUMMARY_PROMPT),
        )
        system_dict = data.get("system")
        if system_dict:
            mem._system = Message(
                role=system_dict["role"],
                content=system_dict["content"],
            )
        mem._summary = data.get("summary")
        mem._messages = cls._dicts_to_messages(data.get("messages", []))
        return mem

    # ------------------------------------------------------------------
    # Propriétés utiles
    # ------------------------------------------------------------------

    @property
    def has_summary(self) -> bool:
        return self._summary is not None

    @property
    def total_stored(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"SummaryMemory("
            f"max={self.max_messages}, "
            f"stored={self.total_stored}, "
            f"has_summary={self.has_summary}, "
            f"llm={'yes' if self.llm else 'no'})"
        )
