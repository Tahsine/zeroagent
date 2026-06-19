"""
core/tools.py — Décorateur @tool + ToolRegistry.

Génère automatiquement un JSON Schema OpenAI-compatible depuis
la signature Python d'une fonction. Zéro dépendance externe :
inspect + typing + functools uniquement.

Usage:
    from zeroagent.core.tools import tool, ToolRegistry

    @tool(description="Recherche des informations sur le web")
    def search(query: str, max_results: int = 5) -> str:
        return f"Résultats pour : {query}"

    registry = ToolRegistry()
    registry.register(search)

    # Pour LLMClient.complete(tools=...)
    schemas = registry.get_schemas()

    # Pour exécuter après un tool_call du LLM
    result = registry.execute("search", {"query": "capitale Bénin"})
"""

import functools
import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin


# ---------------------------------------------------------------------------
# Mapping Python types → JSON Schema
# ---------------------------------------------------------------------------

# Types primitifs directs
_PRIMITIVE_MAP: dict[Any, dict] = {
    str:   {"type": "string"},
    int:   {"type": "integer"},
    float: {"type": "number"},
    bool:  {"type": "boolean"},
    bytes: {"type": "string", "format": "byte"},
}


def _py_type_to_json_schema(py_type: Any) -> dict:
    """
    Convertit un type hint Python en JSON Schema dict.
    Récursif pour les types génériques (list[str], Optional[int]...).

    Couvre :
        str, int, float, bool, bytes
        list, list[T]
        dict, dict[str, V]
        Optional[T]  (= Union[T, None])
        Union[A, B]
        Any
        None / NoneType  → {"type": "null"}
    """
    # Cas Any → pas de contrainte
    if py_type is Any or py_type is inspect.Parameter.empty:
        return {}

    # Cas None / NoneType
    if py_type is type(None):
        return {"type": "null"}

    # Types primitifs directs
    if py_type in _PRIMITIVE_MAP:
        return _PRIMITIVE_MAP[py_type]

    # Types génériques : list, dict, Optional, Union...
    origin = get_origin(py_type)
    args   = get_args(py_type)

    # Optional[T] = Union[T, None]
    # Supporte aussi la syntaxe Python 3.10+ : str | int → types.UnionType
    is_union = origin is typing.Union or isinstance(py_type, types.UnionType)
    if is_union:
        union_args = get_args(py_type)
        non_none = [a for a in union_args if a is not type(None)]
        if len(non_none) == 1:
            # C'est un Optional[T] — on retourne le schema de T
            return _py_type_to_json_schema(non_none[0])
        # Union réel avec plusieurs types non-None
        return {"anyOf": [_py_type_to_json_schema(a) for a in non_none]}

    # list / List[T]
    if origin is list:
        if args:
            return {"type": "array", "items": _py_type_to_json_schema(args[0])}
        return {"type": "array"}

    # dict / Dict[K, V]
    if origin is dict:
        if len(args) >= 2:
            return {
                "type": "object",
                "additionalProperties": _py_type_to_json_schema(args[1]),
            }
        return {"type": "object"}

    # Fallback : bare list / dict sans paramètre (Python 3.9+)
    if py_type is list:
        return {"type": "array"}
    if py_type is dict:
        return {"type": "object"}

    # Dernier recours : string
    return {"type": "string"}


# ---------------------------------------------------------------------------
# Construction du JSON Schema complet depuis une fonction
# ---------------------------------------------------------------------------

def _build_schema_from_fn(fn: Callable, description: str) -> dict:
    """
    Inspecte la signature de `fn` et retourne un JSON Schema
    au format OpenAI tool definition.

    Règles :
    - Les paramètres sans valeur par défaut → required
    - Les paramètres avec défaut (ou Optional) → non required
    - Le paramètre `self` est ignoré
    - Les annotations manquantes → {}  (pas de contrainte de type)

    Exemple de sortie :
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Recherche sur le web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            }
        }
    }
    """
    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Générer le schema du type
        annotation = param.annotation
        prop_schema = _py_type_to_json_schema(annotation)

        # Ajouter la description depuis le docstring si possible
        # (on extrait les sections "Args:" du docstring plus tard)
        properties[name] = prop_schema

        # Required si pas de défaut ET pas Optional
        has_default = param.default is not inspect.Parameter.empty
        origin_ann  = get_origin(annotation)
        args_ann    = get_args(annotation)
        is_optional = (
            # typing.Optional[T] = Union[T, None]
            (origin_ann is typing.Union and type(None) in args_ann)
            # Python 3.10+ : str | None
            or (isinstance(annotation, types.UnionType) and type(None) in get_args(annotation))
        )
        if not has_default and not is_optional:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description or (fn.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# ToolSchema — données d'un tool enregistré
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """Métadonnées d'un tool : son schema JSON + sa fonction callable."""
    name: str
    description: str
    schema: dict          # JSON Schema complet au format OpenAI
    fn: Callable          # la vraie fonction à appeler

    def __call__(self, **kwargs) -> Any:
        return self.fn(**kwargs)


# ---------------------------------------------------------------------------
# Décorateur @tool
# ---------------------------------------------------------------------------

def tool(
    description: str = "",
    name: str | None = None,
):
    """
    Décorateur qui marque une fonction comme outil LLM.

    Attache un attribut `__tool_schema__` (ToolSchema) à la fonction.
    La fonction reste appelable normalement.

    Usage simple :
        @tool(description="Recherche sur le web")
        def search(query: str) -> str: ...

    Usage avec nom custom :
        @tool(description="Calcule une expression", name="calculator")
        def calc(expression: str) -> float: ...

    Usage sans arguments (description depuis le docstring) :
        @tool
        def my_tool(x: int) -> str:
            '''Fait quelque chose d'utile.'''
            ...
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]

        schema = _build_schema_from_fn(fn, tool_desc)
        # Corriger le nom si on a fourni un nom custom
        schema["function"]["name"] = tool_name

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        # Attacher les métadonnées au wrapper
        wrapper.__tool_schema__ = ToolSchema(   # type: ignore[attr-defined]
            name=tool_name,
            description=tool_desc,
            schema=schema,
            fn=fn,
        )
        return wrapper

    # Supporter @tool sans parenthèses (décorateur direct sur une fonction)
    if callable(description):
        fn = description
        description = ""
        return decorator(fn)

    return decorator


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registre de tools. Stocke les ToolSchema et permet l'exécution par nom.

    Usage:
        registry = ToolRegistry()
        registry.register(search)
        registry.register(calculator)

        # Pour LLMClient
        schemas = registry.get_schemas()

        # Après un tool_call du LLM
        result = registry.execute("search", {"query": "Bénin"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}

    def register(self, fn: Callable) -> "ToolRegistry":
        """
        Enregistre un tool décoré avec @tool.
        Lève ValueError si la fonction n'est pas décorée.
        Retourne self pour le chaining.
        """
        schema: ToolSchema | None = getattr(fn, "__tool_schema__", None)
        if schema is None:
            raise ValueError(
                f"La fonction '{fn.__name__}' n'est pas décorée avec @tool. "
                f"Utilise @tool(description='...') avant de l'enregistrer."
            )
        self._tools[schema.name] = schema
        return self

    def register_many(self, *fns: Callable) -> "ToolRegistry":
        """Enregistre plusieurs tools d'un coup."""
        for fn in fns:
            self.register(fn)
        return self

    def get_schemas(self) -> list[dict]:
        """
        Retourne la liste des JSON Schema au format OpenAI.
        À passer directement à LLMClient.complete(tools=...).
        """
        return [ts.schema for ts in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """
        Exécute un tool par son nom avec les arguments fournis.
        Retourne toujours une str (observation pour le harness).

        Lève KeyError si le tool n'existe pas.
        Lève RuntimeError si l'exécution échoue.
        """
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "aucun"
            raise KeyError(
                f"Tool '{name}' introuvable. "
                f"Tools disponibles : {available}"
            )

        tool_schema = self._tools[name]
        try:
            result = tool_schema.fn(**arguments)
            # Toujours retourner une str — le harness l'injecte comme observation
            return str(result) if result is not None else ""
        except TypeError as e:
            raise RuntimeError(
                f"Erreur d'arguments pour le tool '{name}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de l'exécution du tool '{name}': {e}"
            ) from e

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        names = list(self._tools.keys())
        return f"ToolRegistry(tools={names})"
