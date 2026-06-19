"""
tests/test_memory.py

Tests unitaires pour core/memory.py.
Zéro dépendance : unittest uniquement.
"""

import json
import unittest

from zeroagent.core.llm import Message
from zeroagent.core.memory import BufferMemory, WindowMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int, role: str = "user") -> list[Message]:
    return [Message(role=role, content=f"Message {i}") for i in range(n)]

def _sys() -> Message:
    return Message(role="system", content="Tu es un assistant utile.")

def _user(content: str = "Bonjour") -> Message:
    return Message(role="user", content=content)

def _assistant(content: str = "Bonjour !") -> Message:
    return Message(role="assistant", content=content)


# ---------------------------------------------------------------------------
# Tests : BufferMemory
# ---------------------------------------------------------------------------

class TestBufferMemory(unittest.TestCase):

    def test_empty_on_init(self):
        mem = BufferMemory()
        self.assertEqual(mem.get(), [])
        self.assertEqual(len(mem), 0)

    def test_add_and_get(self):
        mem = BufferMemory()
        mem.add(_sys())
        mem.add(_user("Bonjour"))
        mem.add(_assistant("Bonjour !"))

        messages = mem.get()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[2].role, "assistant")

    def test_get_returns_copy(self):
        """Modifier la liste retournée ne doit pas affecter la mémoire."""
        mem = BufferMemory()
        mem.add(_user("test"))
        result = mem.get()
        result.clear()
        self.assertEqual(len(mem), 1)

    def test_clear_keeps_system(self):
        mem = BufferMemory()
        mem.add(_sys())
        mem.add(_user())
        mem.add(_assistant())
        mem.clear()

        messages = mem.get()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "system")

    def test_clear_without_system(self):
        mem = BufferMemory()
        mem.add(_user())
        mem.add(_assistant())
        mem.clear()
        self.assertEqual(mem.get(), [])

    def test_add_many_messages(self):
        mem = BufferMemory()
        for i in range(100):
            mem.add(_user(f"Message {i}"))
        self.assertEqual(len(mem), 100)

    def test_repr(self):
        mem = BufferMemory()
        mem.add(_user())
        r = repr(mem)
        self.assertIn("BufferMemory", r)
        self.assertIn("1", r)

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def test_to_dict(self):
        mem = BufferMemory()
        mem.add(_sys())
        mem.add(_user("test"))
        d = mem.to_dict()

        self.assertEqual(d["type"], "BufferMemory")
        self.assertIn("messages", d)
        self.assertEqual(len(d["messages"]), 2)

    def test_from_dict_roundtrip(self):
        mem = BufferMemory()
        mem.add(_sys())
        mem.add(_user("Question"))
        mem.add(_assistant("Réponse"))

        d = mem.to_dict()
        restored = BufferMemory.from_dict(d)

        self.assertEqual(len(restored.get()), 3)
        self.assertEqual(restored.get()[1].content, "Question")

    def test_to_json_roundtrip(self):
        mem = BufferMemory()
        mem.add(_sys())
        mem.add(_user("JSON test"))

        json_str = mem.to_json()
        self.assertIsInstance(json_str, str)

        restored = BufferMemory.from_json(json_str)
        self.assertEqual(len(restored.get()), 2)
        self.assertEqual(restored.get()[1].content, "JSON test")

    def test_from_dict_empty(self):
        mem = BufferMemory.from_dict({"type": "BufferMemory", "messages": []})
        self.assertEqual(mem.get(), [])

    def test_tool_message_roundtrip(self):
        """Les messages tool avec tool_call_id doivent survivre à la sérialisation."""
        mem = BufferMemory()
        mem.add(Message(role="tool", content="Paris", tool_call_id="call_001", name="search"))
        d = mem.to_dict()
        restored = BufferMemory.from_dict(d)
        msg = restored.get()[0]
        self.assertEqual(msg.role, "tool")
        self.assertEqual(msg.tool_call_id, "call_001")
        self.assertEqual(msg.name, "search")


# ---------------------------------------------------------------------------
# Tests : WindowMemory
# ---------------------------------------------------------------------------

class TestWindowMemory(unittest.TestCase):

    def test_init_default_k(self):
        mem = WindowMemory()
        self.assertEqual(mem.k, 20)

    def test_init_custom_k(self):
        mem = WindowMemory(k=5)
        self.assertEqual(mem.k, 5)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            WindowMemory(k=0)
        with self.assertRaises(ValueError):
            WindowMemory(k=-1)

    def test_empty_on_init(self):
        mem = WindowMemory(k=5)
        self.assertEqual(mem.get(), [])

    def test_system_stored_separately(self):
        """Le system prompt est stocké à part et réinjecté en tête."""
        mem = WindowMemory(k=3)
        mem.add(_sys())
        mem.add(_user("1"))
        mem.add(_user("2"))

        messages = mem.get()
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(len(messages), 3)  # system + 2 messages

    def test_window_limits_messages(self):
        """Avec k=3, get() ne retourne que les 3 derniers messages non-system."""
        mem = WindowMemory(k=3)
        for i in range(10):
            mem.add(_user(f"Message {i}"))

        messages = mem.get()
        self.assertEqual(len(messages), 3)
        # Les 3 derniers
        self.assertEqual(messages[0].content, "Message 7")
        self.assertEqual(messages[1].content, "Message 8")
        self.assertEqual(messages[2].content, "Message 9")

    def test_window_with_system_limits_correctly(self):
        """Avec system + k=3 : get() retourne system + 3 derniers."""
        mem = WindowMemory(k=3)
        mem.add(_sys())
        for i in range(10):
            mem.add(_user(f"Message {i}"))

        messages = mem.get()
        self.assertEqual(len(messages), 4)  # system + 3
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].content, "Message 7")

    def test_system_always_preserved(self):
        """Le system prompt doit survivre même après beaucoup de messages."""
        mem = WindowMemory(k=2)
        mem.add(_sys())
        for i in range(100):
            mem.add(_user(f"Message {i}"))

        messages = mem.get()
        self.assertEqual(messages[0].role, "system")

    def test_system_replaced_if_added_again(self):
        """Un nouveau message system remplace l'ancien."""
        mem = WindowMemory(k=5)
        mem.add(Message(role="system", content="Premier système"))
        mem.add(Message(role="system", content="Nouveau système"))

        messages = mem.get()
        system_messages = [m for m in messages if m.role == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0].content, "Nouveau système")

    def test_clear_keeps_system(self):
        mem = WindowMemory(k=5)
        mem.add(_sys())
        mem.add(_user())
        mem.add(_assistant())
        mem.clear()

        messages = mem.get()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "system")

    def test_clear_without_system(self):
        mem = WindowMemory(k=5)
        mem.add(_user())
        mem.add(_assistant())
        mem.clear()
        self.assertEqual(mem.get(), [])

    def test_total_stored(self):
        """total_stored compte les messages non-system accumulés."""
        mem = WindowMemory(k=3)
        mem.add(_sys())
        for i in range(7):
            mem.add(_user(f"msg {i}"))
        self.assertEqual(mem.total_stored, 7)

    def test_is_full(self):
        mem = WindowMemory(k=3)
        self.assertFalse(mem.is_full)
        mem.add(_user("1"))
        mem.add(_user("2"))
        self.assertFalse(mem.is_full)
        mem.add(_user("3"))
        self.assertTrue(mem.is_full)

    def test_within_window_returns_all(self):
        """Si moins de k messages, tous sont retournés."""
        mem = WindowMemory(k=10)
        mem.add(_user("A"))
        mem.add(_assistant("B"))
        self.assertEqual(len(mem.get()), 2)

    def test_repr(self):
        mem = WindowMemory(k=5)
        mem.add(_sys())
        r = repr(mem)
        self.assertIn("WindowMemory", r)
        self.assertIn("k=5", r)

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def test_to_dict(self):
        mem = WindowMemory(k=5)
        mem.add(_sys())
        mem.add(_user("test"))
        d = mem.to_dict()

        self.assertEqual(d["type"], "WindowMemory")
        self.assertEqual(d["k"], 5)
        self.assertIsNotNone(d["system"])

    def test_from_dict_roundtrip(self):
        mem = WindowMemory(k=4)
        mem.add(_sys())
        mem.add(_user("Question"))
        mem.add(_assistant("Réponse"))

        d = mem.to_dict()
        restored = WindowMemory.from_dict(d)

        self.assertEqual(restored.k, 4)
        messages = restored.get()
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].content, "Question")

    def test_from_dict_without_system(self):
        mem = WindowMemory(k=5)
        mem.add(_user("Sans système"))
        d = mem.to_dict()
        restored = WindowMemory.from_dict(d)

        self.assertIsNone(restored._system)
        self.assertEqual(len(restored.get()), 1)

    def test_to_json_roundtrip(self):
        mem = WindowMemory(k=3)
        mem.add(_sys())
        mem.add(_user("JSON test"))

        json_str = mem.to_json()
        restored = WindowMemory.from_json(json_str)

        self.assertEqual(restored.k, 3)
        self.assertEqual(restored.get()[1].content, "JSON test")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
