"""
workflow/state.py — State partagé entre les nœuds d'un Graph.

State est un dict wrappé avec :
- accès par clé (state["key"] ou state.get("key"))
- merge explicite (state.merge(other_dict))
- copie propre entre nœuds pour éviter les mutations silencieuses
- validation légère via __annotations__ si une sous-classe est définie
- sérialisation JSON pour debug / logging

Design :
  - Pas de TypedDict ni de Pydantic — stdlib uniquement
  - Pas de channels / réduction automatique — le nœud merge ce qu'il veut
  - La sous-classe optionnelle permet de documenter le schéma attendu :

    class MyState(State):
        query: str
        result: str = ""
        iterations: int = 0

    s = MyState({"query": "test"})
    # Valeurs par défaut injectées depuis les annotations
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterator


class State:
    """
    Conteneur de données partagé entre les nœuds d'un Graph.

    Wrapping explicite d'un dict avec merge, copie, et validation légère.
    Sous-classable pour documenter le schéma sans Pydantic.

    Usage basique :
        state = State({"query": "test", "result": ""})
        state["result"] = "done"
        print(state.get("result"))   # "done"
        print("query" in state)      # True

    Sous-classé avec schéma :
        class PipelineState(State):
            query: str
            result: str = ""
            retries: int = 0

        state = PipelineState({"query": "test"})
        # result et retries sont injectés avec leurs valeurs par défaut
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}

        # Injecter les valeurs par défaut depuis les annotations de la sous-classe
        if type(self) is not State:
            self._inject_defaults()

        # Appliquer les données fournies (écrasent les défauts)
        if data:
            self._data.update(data)

    # ------------------------------------------------------------------
    # Accès dict-like
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def __repr__(self) -> str:
        cls = type(self).__name__
        keys = list(self._data.keys())
        return f"{cls}({keys})"

    # ------------------------------------------------------------------
    # API State
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retourne la valeur ou default si la clé est absente."""
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, data: dict[str, Any]) -> None:
        """Met à jour plusieurs clés en une fois."""
        self._data.update(data)

    def merge(self, data: dict[str, Any]) -> "State":
        """
        Merge un dict dans le State et retourne self (chainable).
        Différence avec update() : retourne self pour le chaining.

        Usage dans un nœud :
            def my_node(state: State) -> State:
                return state.merge({"result": "done", "step": 2})
        """
        self._data.update(data)
        return self

    def copy(self) -> "State":
        """
        Retourne une copie profonde du State.
        Utilisé par Graph pour passer des snapshots entre nœuds
        sans risquer des mutations silencieuses.
        """
        new = type(self).__new__(type(self))
        new._data = copy.deepcopy(self._data)
        return new

    def to_dict(self) -> dict[str, Any]:
        """Retourne le dict sous-jacent (copie superficielle)."""
        return dict(self._data)

    def to_json(self, indent: int = 2) -> str:
        """Sérialise en JSON pour debug / logging."""
        def _default(obj):
            return repr(obj)
        return json.dumps(self._data, ensure_ascii=False, indent=indent, default=_default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        """Constructeur alternatif depuis un dict."""
        return cls(data)

    # ------------------------------------------------------------------
    # Validation légère (sous-classes uniquement)
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Vérifie que les clés requises (annotations sans valeur par défaut)
        sont présentes dans le State.

        Retourne une liste d'erreurs (vide si tout est OK).
        Ne lève pas d'exception — laisse le Graph décider quoi faire.

        Usage :
            errors = state.validate()
            if errors:
                raise ValueError(f"State invalide: {errors}")
        """
        errors: list[str] = []
        cls = type(self)
        if cls is State:
            return errors

        annotations = {}
        for klass in reversed(cls.__mro__):
            annotations.update(getattr(klass, "__annotations__", {}))

        defaults = {}
        for klass in cls.__mro__:
            for k, v in vars(klass).items():
                if not k.startswith("_") and k in annotations:
                    defaults[k] = v

        for key in annotations:
            if key.startswith("_"):
                continue
            if key not in defaults and key not in self._data:
                errors.append(f"Clé requise manquante: '{key}'")

        return errors

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _inject_defaults(self) -> None:
        """
        Injecte les valeurs par défaut depuis les annotations de la sous-classe.
        Appelé uniquement dans __init__ pour les sous-classes.
        """
        cls = type(self)
        annotations: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            annotations.update(getattr(klass, "__annotations__", {}))

        for key in annotations:
            if key.startswith("_"):
                continue
            # Chercher une valeur par défaut dans la hiérarchie de classes
            for klass in cls.__mro__:
                if key in vars(klass) and not callable(vars(klass)[key]):
                    self._data[key] = copy.deepcopy(vars(klass)[key])
                    break
