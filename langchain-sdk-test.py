"""
langchain-sdk-test.py — Test comparatif zeroagent vs LangChain create_agent.

Mêmes flags CLI, mêmes couleurs, même tool calculator, même .env.
Utilise ChatOpenAI avec base_url vers Ollama cloud (API OpenAI-compatible).

Usage:
  python langchain-sdk-test.py "Calcule 123 * 456"
  python langchain-sdk-test.py --chat
  python langchain-sdk-test.py --chat --stream
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from typing import NoReturn

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------------------------
# Couleurs ANSI — copié de zeroagent __main__.py
# ---------------------------------------------------------------------------

class _Colors:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def dim(self, t: str) -> str:      return self("2", t)
    def bold(self, t: str) -> str:     return self("1", t)
    def cyan(self, t: str) -> str:     return self("36", t)
    def green(self, t: str) -> str:    return self("32", t)
    def yellow(self, t: str) -> str:   return self("33", t)
    def red(self, t: str) -> str:      return self("31", t)
    def magenta(self, t: str) -> str:  return self("35", t)


_use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
C = _Colors(_use_color)


# ---------------------------------------------------------------------------
# Loader .env — copié de zeroagent __main__.py
# ---------------------------------------------------------------------------

def load_env(*paths: str) -> str | None:
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return path
    return None


# ---------------------------------------------------------------------------
# Calculator tool — strictement identique à zeroagent
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Effectue un calcul mathématique : +, -, *, /, parenthèses"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Expression non autorisée : {expression}"
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Erreur de calcul : {e}"


def _strip_think_tags(text: str) -> str:
    """Supprime les blocs <think>...</think> du texte."""
    return re.sub(r"\s*<think>.*?</think>\s*", " ", text, flags=re.DOTALL).strip()


DEFAULT_SYSTEM_PROMPT = "Tu es un assistant utile et concis. Tu réponds en français."


# ---------------------------------------------------------------------------
# Construction du LLM + agent
# ---------------------------------------------------------------------------

def _fail(msg: str) -> NoReturn:
    print(C.red(f"Erreur : {msg}"), file=sys.stderr)
    sys.exit(1)


def build_llm(args: argparse.Namespace) -> ChatOpenAI:
    base_url = args.base_url or os.environ.get("LLM_BASE_URL")
    model    = args.model or os.environ.get("LLM_MODEL")
    api_key  = args.api_key or os.environ.get("LLM_API_KEY", "")

    if not base_url:
        _fail("LLM_BASE_URL manquant. Définis-le dans .env ou passe --base-url.")
    if not model:
        _fail("LLM_MODEL manquant. Définis-le dans .env ou passe --model.")

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def build_agent(llm: ChatOpenAI, args: argparse.Namespace) -> "CompiledStateGraph":
    tools_list = [] if args.no_tools else [calculator]
    agent = create_agent(
        model=llm,
        tools=tools_list,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# ---------------------------------------------------------------------------
# Verbose — parcourt les messages pour afficher Thought/Action/Observation
# ---------------------------------------------------------------------------

def _print_messages_verbose(messages: list) -> None:
    """Affiche les tool calls et résultats comme zeroagent --verbose."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            short = _strip_think_tags(msg.content).replace("\n", " ")
            if len(short) > 100:
                short = short[:100] + "…"
            print(C.dim(f"  · {short}"))
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(C.cyan(f"  → {tc['name']}") + C.dim(f"({tc['args']})"))
        elif isinstance(msg, ToolMessage):
            short = msg.content.strip().replace("\n", " ")
            if len(short) > 100:
                short = short[:100] + "…"
            print(f"  {C.green('✓')} {C.dim(short)}")


# ---------------------------------------------------------------------------
# Mode one-shot
# ---------------------------------------------------------------------------

def run_once(args: argparse.Namespace) -> None:
    llm   = build_llm(args)
    agent = build_agent(llm, args)

    if not args.quiet:
        tools_str = [calculator.name] if not args.no_tools else []
        print(C.dim(f"model={llm.model}  tools={tools_str}\n"))

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": args.max_iterations * 3 + 10,
    }

    inputs = {"messages": [HumanMessage(content=args.question)]}
    result = agent.invoke(inputs, config=config)
    messages = result["messages"]

    if args.verbose and not args.quiet:
        _print_messages_verbose(messages)

    # Dernier message = réponse finale
    last = messages[-1]
    answer = _strip_think_tags(last.content) if hasattr(last, "content") else str(last)

    if args.quiet:
        print(answer)
    else:
        print()
        print(C.bold(C.green("Réponse : ")) + answer)

        # Compter les tool calls
        tool_calls = sum(
            1 for m in messages
            if isinstance(m, AIMessage) and m.tool_calls
        )
        print(C.dim(f"\n[1 run · {tool_calls} tool call(s)]"))


# ---------------------------------------------------------------------------
# Mode chat REPL
# ---------------------------------------------------------------------------

def run_chat(args: argparse.Namespace) -> None:
    llm = build_llm(args)

    print(C.bold("langchain-sdk-test") + C.dim(" — mode chat interactif"))
    print(C.dim(f"model={llm.model_name}  base_url={llm.openai_api_base or 'default'}"))
    print(C.dim("Commandes : /reset  /tools  /quit  (ou Ctrl+D)\n"))

    if args.stream:
        _run_chat_streaming(llm, args)
    else:
        _run_chat_agent(llm, args)


def _run_chat_agent(llm: ChatOpenAI, args: argparse.Namespace) -> None:
    agent = create_agent(
        model=llm,
        tools=[] if args.no_tools else [calculator],
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input(C.bold(C.cyan("\nVous › ")))
        except (EOFError, KeyboardInterrupt):
            print(C.dim("\nÀ bientôt."))
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            print(C.dim("À bientôt."))
            break
        if user_input == "/reset":
            thread_id = str(uuid.uuid4())
            print(C.dim("Mémoire réinitialisée."))
            continue
        if user_input == "/tools":
            tools_str = [calculator.name] if not args.no_tools else []
            print(C.dim(f"Tools : {tools_str}"))
            continue

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": args.max_iterations * 3 + 10,
        }
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = agent.invoke(inputs, config=config)

        if args.verbose:
            _print_messages_verbose(result["messages"])

        last = result["messages"][-1]
        answer = _strip_think_tags(last.content) if hasattr(last, "content") else str(last)
        print(C.bold(C.green("\nAgent › ")) + answer)


def _run_chat_streaming(llm: ChatOpenAI, args: argparse.Namespace) -> None:
    history: list = []
    if args.system_prompt:
        from langchain_core.messages import SystemMessage
        history.append(SystemMessage(content=args.system_prompt))

    while True:
        try:
            user_input = input(C.bold(C.cyan("\nVous › ")))
        except (EOFError, KeyboardInterrupt):
            print(C.dim("\nÀ bientôt."))
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            print(C.dim("À bientôt."))
            break
        if user_input == "/reset":
            history = [m for m in history if isinstance(m, SystemMessage)]
            print(C.dim("Mémoire réinitialisée."))
            continue

        history.append(HumanMessage(content=user_input))

        print(C.bold(C.green("\nAgent › ")), end="", flush=True)
        full_response = ""
        try:
            for chunk in llm.stream(history):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                clean = _strip_think_tags(content)
                print(clean, end="", flush=True)
                full_response += content  # garder le raw dans l'history pour le contexte
        except Exception as e:
            print(C.red(f"\n[erreur stream] {e}"))
            history.pop()
            continue
        print()

        history.append(AIMessage(content=full_response))


# ---------------------------------------------------------------------------
# Argparse — flags identiques à zeroagent
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langchain-sdk-test",
        description="Test LangChain create_agent — comparaison avec zeroagent.",
    )
    parser.add_argument("--base-url", default=None, help="Override LLM_BASE_URL")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL")
    parser.add_argument("--api-key", default=None, help="Override LLM_API_KEY")
    parser.add_argument("--system-prompt", default=None, help="System prompt custom")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--no-tools", action="store_true", help="Désactive les tools par défaut")
    parser.add_argument("--verbose", action="store_true", help="Affiche Thought/Action/Observation")
    parser.add_argument("--quiet", action="store_true", help="Sortie minimale (juste la réponse)")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit code 1 si le run échoue")
    parser.add_argument("--env", default=None, help="Chemin custom vers un fichier .env")

    parser.add_argument("--chat", action="store_true", help="Mode chat interactif (REPL)")
    parser.add_argument("--stream", action="store_true", help="Streaming token par token (mode --chat uniquement, désactive les tools)")
    parser.add_argument("question", nargs="?", help="Question pour le mode one-shot")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.env:
        env_path = load_env(args.env)
    else:
        env_path = load_env(".env", "zeroagent/.env", "../.env")

    if env_path and not args.quiet:
        print(C.dim(f"[env] {env_path}"))

    if args.chat:
        run_chat(args)
        return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    run_once(args)


if __name__ == "__main__":
    main()
