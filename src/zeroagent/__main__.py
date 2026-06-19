"""
zeroagent CLI — point d'entrée `python -m zeroagent` ou `zeroagent` (après pip install).

Trois modes :
  zeroagent "question"          → one-shot, réponse unique
  zeroagent chat                → REPL interactif avec mémoire de session
  zeroagent chat --stream       → REPL avec streaming token par token (sans tools)

Configuration via .env (auto-détecté) ou variables d'environnement :
  LLM_BASE_URL   ex: https://ollama.com/v1
  LLM_MODEL      ex: minimax-m3:cloud
  LLM_API_KEY    ta clé API (vide si non requis, ex: Ollama local)

Zéro dépendance externe : argparse + stdlib uniquement.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

from zeroagent.core.llm import LLMClient, Message
from zeroagent.core.tools import tool
from zeroagent.harness.agent import Agent


# ---------------------------------------------------------------------------
# Couleurs ANSI — désactivées si pas un TTY ou si NO_COLOR est défini
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
# Loader .env — stdlib pure, cherche dans plusieurs emplacements
# ---------------------------------------------------------------------------

def load_env(*paths: str) -> str | None:
    """Charge le premier .env trouvé. Retourne le chemin chargé ou None."""
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
# Tools de démo par défaut — utilisés si --no-tools n'est pas passé
# ---------------------------------------------------------------------------

@tool(description="Effectue un calcul mathématique : +, -, *, /, parenthèses")
def calculator(expression: str) -> str:
    """Évalue une expression arithmétique simple."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Expression non autorisée : {expression}"
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Erreur de calcul : {e}"


DEFAULT_SYSTEM_PROMPT = """Tu es un assistant utile et concis. Tu réponds en français.

Quand tu as suffisamment d'informations pour répondre, réponds IMMÉDIATEMENT
avec ce format exact :

Final Answer: [ta réponse complète ici]

N'utilise les outils que si nécessaire. Dès que tu as la réponse, utilise Final Answer."""


# ---------------------------------------------------------------------------
# Construction du client + agent depuis les args / env
# ---------------------------------------------------------------------------

def _fail(msg: str) -> NoReturn:
    print(C.red(f"Erreur : {msg}"), file=sys.stderr)
    sys.exit(1)


def build_llm(args: argparse.Namespace) -> LLMClient:
    base_url = args.base_url or os.environ.get("LLM_BASE_URL")
    model    = args.model or os.environ.get("LLM_MODEL")
    api_key  = args.api_key or os.environ.get("LLM_API_KEY", "")

    if not base_url:
        _fail("LLM_BASE_URL manquant. Définis-le dans .env ou passe --base-url.")
    if not model:
        _fail("LLM_MODEL manquant. Définis-le dans .env ou passe --model.")

    return LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def build_agent(llm: LLMClient, args: argparse.Namespace) -> Agent:
    tools = [] if args.no_tools else [calculator]
    # verbose=False ici : on utilise nos propres hooks pretty pour l'affichage
    # (voir _attach_pretty_hooks) plutôt que les print() bruts du RunConfig
    return Agent(
        llm=llm,
        tools=tools,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_iterations=args.max_iterations,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Affichage des hooks — Thought/Action/Observation en couleur
# ---------------------------------------------------------------------------

def _attach_pretty_hooks(agent: Agent) -> None:
    """Remplace les hooks internes pour un affichage joli pendant la boucle."""

    def on_thought(thought: str, iteration: int) -> None:
        short = thought.strip().replace("\n", " ")
        if len(short) > 100:
            short = short[:100] + "…"
        print(C.dim(f"  · {short}"))

    def on_action(name: str, args: dict, iteration: int) -> None:
        print(C.cyan(f"  → {name}") + C.dim(f"({args})"))

    def on_observation(name: str, obs: str, success: bool, iteration: int) -> None:
        short = obs.strip().replace("\n", " ")
        if len(short) > 100:
            short = short[:100] + "…"
        marker = C.green("✓") if success else C.red("✗")
        print(f"  {marker} {C.dim(short)}")

    agent._loop._on_thought     = on_thought
    agent._loop._on_action      = on_action
    agent._loop._on_observation = on_observation


# ---------------------------------------------------------------------------
# Mode one-shot
# ---------------------------------------------------------------------------

def run_once(args: argparse.Namespace) -> None:
    llm   = build_llm(args)
    agent = build_agent(llm, args)
    if args.verbose and not args.quiet:
        _attach_pretty_hooks(agent)
    if not args.quiet:
        print(C.dim(f"model={llm.model}  tools={agent.tools}\n"))

    result = agent.run_full(args.question)

    if args.quiet:
        print(result.answer)
    else:
        print()
        print(C.bold(C.green("Réponse : ")) + result.answer)
        print(C.dim(
            f"\n[{result.iterations} itération(s) · "
            f"{result.tool_calls_made} tool call(s) · "
            f"{result.stop_reason.value}]"
        ))

    if not result.success and args.fail_on_error:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mode chat REPL
# ---------------------------------------------------------------------------

def run_chat(args: argparse.Namespace) -> None:
    llm = build_llm(args)

    print(C.bold("zeroagent") + C.dim(" — mode chat interactif"))
    print(C.dim(f"model={llm.model}  base_url={llm.base_url}"))
    print(C.dim("Commandes : /reset  /tools  /quit  (ou Ctrl+D)\n"))

    if args.stream:
        # Mode streaming pur : pas de tools, juste un chat simple
        _run_chat_streaming(llm, args)
    else:
        _run_chat_agent(llm, args)


def _run_chat_agent(llm: LLMClient, args: argparse.Namespace) -> None:
    agent = build_agent(llm, args)
    if args.verbose:
        _attach_pretty_hooks(agent)

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
            agent.reset()
            print(C.dim("Mémoire réinitialisée."))
            continue
        if user_input == "/tools":
            print(C.dim(f"Tools : {agent.tools}"))
            continue

        result = agent.run_full(user_input)
        print(C.bold(C.green("\nAgent › ")) + result.answer)
        if args.verbose:
            print(C.dim(
                f"[{result.iterations} itération(s) · "
                f"{result.tool_calls_made} tool call(s) · "
                f"{result.stop_reason.value}]"
            ))


def _run_chat_streaming(llm: LLMClient, args: argparse.Namespace) -> None:
    """
    Chat avec streaming token par token via LLMClient.stream().
    Pas de tool calling ici — c'est un chat texte simple,
    la boucle ReAct a besoin du texte complet pour parser une action.
    """
    history: list[Message] = []
    if args.system_prompt:
        history.append(Message(role="system", content=args.system_prompt))

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
            system = [m for m in history if m.role == "system"]
            history = system
            print(C.dim("Mémoire réinitialisée."))
            continue

        history.append(Message(role="user", content=user_input))

        print(C.bold(C.green("\nAgent › ")), end="", flush=True)
        full_response = ""
        try:
            for chunk in llm.stream(history):
                print(chunk, end="", flush=True)
                full_response += chunk
        except Exception as e:
            print(C.red(f"\n[erreur stream] {e}"))
            history.pop()  # retire le message user qui a échoué
            continue
        print()

        history.append(Message(role="assistant", content=full_response))


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zeroagent",
        description="Agent IA + Workflow SDK en Python pur. Zéro dépendance.",
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
