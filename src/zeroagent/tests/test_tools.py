"""
tests/test_tools.py

Tests unitaires pour core/tools.py.
Zéro dépendance : unittest uniquement.
"""

import unittest
from typing import Any, Optional

from zeroagent.core.tools import (
    ToolRegistry,
    ToolSchema,
    _build_schema_from_fn,
    _py_type_to_json_schema,
    tool,
)


# ---------------------------------------------------------------------------
# Tests : _py_type_to_json_schema
# ---------------------------------------------------------------------------

class TestPyTypeToJsonSchema(unittest.TestCase):

    def test_str(self):
        self.assertEqual(_py_type_to_json_schema(str), {"type": "string"})

    def test_int(self):
        self.assertEqual(_py_type_to_json_schema(int), {"type": "integer"})

    def test_float(self):
        self.assertEqual(_py_type_to_json_schema(float), {"type": "number"})

    def test_bool(self):
        self.assertEqual(_py_type_to_json_schema(bool), {"type": "boolean"})

    def test_bytes(self):
        result = _py_type_to_json_schema(bytes)
        self.assertEqual(result["type"], "string")

    def test_bare_list(self):
        self.assertEqual(_py_type_to_json_schema(list), {"type": "array"})

    def test_list_of_str(self):
        result = _py_type_to_json_schema(list[str])
        self.assertEqual(result, {"type": "array", "items": {"type": "string"}})

    def test_list_of_int(self):
        result = _py_type_to_json_schema(list[int])
        self.assertEqual(result, {"type": "array", "items": {"type": "integer"}})

    def test_bare_dict(self):
        self.assertEqual(_py_type_to_json_schema(dict), {"type": "object"})

    def test_dict_str_str(self):
        result = _py_type_to_json_schema(dict[str, str])
        self.assertEqual(result["type"], "object")
        self.assertIn("additionalProperties", result)

    def test_optional_str(self):
        """Optional[str] doit retourner le schema de str."""
        result = _py_type_to_json_schema(Optional[str])
        self.assertEqual(result, {"type": "string"})

    def test_optional_int(self):
        result = _py_type_to_json_schema(Optional[int])
        self.assertEqual(result, {"type": "integer"})

    def test_any(self):
        """Any doit retourner {} — pas de contrainte."""
        self.assertEqual(_py_type_to_json_schema(Any), {})

    def test_none_type(self):
        self.assertEqual(_py_type_to_json_schema(type(None)), {"type": "null"})

    def test_union_two_types(self):
        """Union[str, int] → anyOf."""
        result = _py_type_to_json_schema(str | int)
        self.assertIn("anyOf", result)
        self.assertEqual(len(result["anyOf"]), 2)


# ---------------------------------------------------------------------------
# Tests : _build_schema_from_fn
# ---------------------------------------------------------------------------

class TestBuildSchemaFromFn(unittest.TestCase):

    def test_simple_function(self):
        def search(query: str) -> str:
            pass

        schema = _build_schema_from_fn(search, "Recherche sur le web")
        fn = schema["function"]

        self.assertEqual(schema["type"], "function")
        self.assertEqual(fn["name"], "search")
        self.assertEqual(fn["description"], "Recherche sur le web")
        self.assertIn("query", fn["parameters"]["properties"])
        self.assertIn("query", fn["parameters"]["required"])

    def test_default_not_required(self):
        """Les paramètres avec défaut ne doivent pas être dans required."""
        def search(query: str, max_results: int = 5) -> str:
            pass

        schema = _build_schema_from_fn(search, "Search")
        required = schema["function"]["parameters"]["required"]

        self.assertIn("query", required)
        self.assertNotIn("max_results", required)

    def test_optional_not_required(self):
        """Optional[X] ne doit pas être dans required."""
        def fn(name: str, alias: Optional[str] = None) -> str:
            pass

        schema = _build_schema_from_fn(fn, "Test")
        required = schema["function"]["parameters"]["required"]

        self.assertIn("name", required)
        self.assertNotIn("alias", required)

    def test_no_annotation_becomes_empty_schema(self):
        """Paramètre sans annotation → {} dans properties."""
        def fn(x) -> str:
            pass

        schema = _build_schema_from_fn(fn, "Test")
        props = schema["function"]["parameters"]["properties"]
        self.assertEqual(props["x"], {})

    def test_self_ignored(self):
        """Le paramètre 'self' doit être ignoré."""
        class MyClass:
            def method(self, query: str) -> str:
                pass

        schema = _build_schema_from_fn(MyClass.method, "Test")
        props = schema["function"]["parameters"]["properties"]

        self.assertNotIn("self", props)
        self.assertIn("query", props)

    def test_description_from_docstring(self):
        """Si description vide, utilise le docstring."""
        def fn(x: int) -> int:
            """Calcule quelque chose."""
            return x

        schema = _build_schema_from_fn(fn, "")
        self.assertIn("Calcule", schema["function"]["description"])

    def test_list_param(self):
        def fn(items: list[str]) -> str:
            pass

        schema = _build_schema_from_fn(fn, "Test")
        prop = schema["function"]["parameters"]["properties"]["items"]
        self.assertEqual(prop["type"], "array")
        self.assertEqual(prop["items"]["type"], "string")


# ---------------------------------------------------------------------------
# Tests : décorateur @tool
# ---------------------------------------------------------------------------

class TestToolDecorator(unittest.TestCase):

    def test_basic_decoration(self):
        @tool(description="Recherche")
        def search(query: str) -> str:
            return f"Résultat: {query}"

        self.assertTrue(hasattr(search, "__tool_schema__"))
        schema: ToolSchema = search.__tool_schema__
        self.assertEqual(schema.name, "search")
        self.assertEqual(schema.description, "Recherche")

    def test_function_still_callable(self):
        """La fonction décorée doit rester appelable normalement."""
        @tool(description="Double")
        def double(x: int) -> int:
            return x * 2

        self.assertEqual(double(5), 10)
        self.assertEqual(double(x=3), 6)

    def test_custom_name(self):
        @tool(description="Calc", name="calculator")
        def calc(expr: str) -> float:
            return 0.0

        schema: ToolSchema = calc.__tool_schema__
        self.assertEqual(schema.name, "calculator")
        self.assertEqual(schema.schema["function"]["name"], "calculator")

    def test_functools_wraps_preserved(self):
        """Le nom et docstring de la fonction originale doivent être préservés."""
        @tool(description="Test")
        def my_function(x: int) -> str:
            """Ma fonction de test."""
            return str(x)

        self.assertEqual(my_function.__name__, "my_function")
        self.assertEqual(my_function.__doc__, "Ma fonction de test.")

    def test_schema_structure(self):
        """Vérifie la structure complète du JSON Schema généré."""
        @tool(description="Recherche avec options")
        def search(query: str, max_results: int = 5, verbose: bool = False) -> str:
            pass

        schema = search.__tool_schema__.schema
        fn = schema["function"]

        self.assertEqual(schema["type"], "function")
        self.assertIn("query", fn["parameters"]["properties"])
        self.assertIn("max_results", fn["parameters"]["properties"])
        self.assertIn("verbose", fn["parameters"]["properties"])

        # Seul query est required
        self.assertEqual(fn["parameters"]["required"], ["query"])

    def test_tool_without_parentheses(self):
        """@tool sans parenthèses doit fonctionner aussi."""
        @tool
        def simple(x: str) -> str:
            """Description depuis docstring."""
            return x

        self.assertTrue(hasattr(simple, "__tool_schema__"))
        self.assertEqual(simple.__tool_schema__.name, "simple")

    def test_tool_schema_fn_is_original(self):
        """Le fn dans ToolSchema doit être la fonction originale."""
        @tool(description="Test")
        def compute(x: int) -> int:
            return x + 1

        result = compute.__tool_schema__.fn(x=10)
        self.assertEqual(result, 11)


# ---------------------------------------------------------------------------
# Tests : ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        @tool(description="Recherche web")
        def search(query: str) -> str:
            return f"Résultats pour: {query}"

        @tool(description="Calcul mathématique")
        def calculator(expression: str) -> str:
            return str(eval(expression))  # noqa: S307 (test only)

        self.search = search
        self.calculator = calculator

    def test_register_tool(self):
        registry = ToolRegistry()
        registry.register(self.search)
        self.assertIn("search", registry)
        self.assertEqual(len(registry), 1)

    def test_register_many(self):
        registry = ToolRegistry()
        registry.register_many(self.search, self.calculator)
        self.assertEqual(len(registry), 2)

    def test_register_non_tool_raises(self):
        registry = ToolRegistry()

        def plain_function(x: str) -> str:
            return x

        with self.assertRaises(ValueError):
            registry.register(plain_function)

    def test_get_schemas_returns_list(self):
        registry = ToolRegistry()
        registry.register_many(self.search, self.calculator)
        schemas = registry.get_schemas()

        self.assertIsInstance(schemas, list)
        self.assertEqual(len(schemas), 2)
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIn("function", s)

    def test_execute_tool(self):
        registry = ToolRegistry()
        registry.register(self.search)
        result = registry.execute("search", {"query": "capitale Bénin"})
        self.assertIsInstance(result, str)
        self.assertIn("capitale Bénin", result)

    def test_execute_unknown_tool_raises(self):
        registry = ToolRegistry()
        with self.assertRaises(KeyError):
            registry.execute("nonexistent", {})

    def test_execute_bad_args_raises(self):
        registry = ToolRegistry()
        registry.register(self.search)
        with self.assertRaises(RuntimeError):
            registry.execute("search", {"wrong_arg": "oops"})

    def test_execute_returns_str(self):
        """execute() doit toujours retourner une str."""
        @tool(description="Retourne un int")
        def returns_int(x: int) -> int:
            return x * 2

        registry = ToolRegistry()
        registry.register(returns_int)
        result = registry.execute("returns_int", {"x": 5})
        self.assertIsInstance(result, str)
        self.assertEqual(result, "10")

    def test_chaining(self):
        """register() doit retourner self pour le chaining."""
        registry = ToolRegistry()
        result = registry.register(self.search)
        self.assertIs(result, registry)

    def test_repr(self):
        registry = ToolRegistry()
        registry.register(self.search)
        r = repr(registry)
        self.assertIn("search", r)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
