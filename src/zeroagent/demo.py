# demo.py — zeroagent Day 1
# Agent IA en Python pur. Zéro dépendance externe.
# github.com/Tahsine/zeroagent

import os
import sys

# ─────────────────────────────────────────────
# 0. Loader .env minimal — stdlib pure
#    Lit .env ou zeroagent/.env automatiquement
# ─────────────────────────────────────────────

def _load_env(*paths: str) -> None:
    """
    Charge un fichier .env dans os.environ.
    Ignore les lignes vides et les commentaires (#).
    N'écrase pas les variables déjà définies dans l'environnement.
    """
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                # Ne pas écraser une variable déjà présente dans l'env
                if key and key not in os.environ:
                    os.environ[key] = value
        print(f"[env] Chargé : {path}")
        break   # On prend le premier fichier trouvé

_load_env(".env", "zeroagent/.env", "../.env")


# ─────────────────────────────────────────────
# 1. Le tool — une fonction Python normale
# ─────────────────────────────────────────────

from zeroagent.core.tools import tool

@tool(description="Recherche des informations sur le Bénin : capitale, population, monnaie, langues")
def search(query: str) -> str:
    """
    Base de connaissances locale sur le Bénin.
    Retourne une réponse complète qui couvre capitale ET population
    quand les deux sont demandés.
    """
    query_lower = query.lower()

    # Réponse combinée si la query mentionne capitale + population/habitants
    wants_capital   = any(w in query_lower for w in ["capitale", "capital"])
    wants_pop       = any(w in query_lower for w in ["population", "habitants", "habitant"])
    wants_currency  = any(w in query_lower for w in ["monnaie", "argent", "cfa", "franc"])
    wants_language  = any(w in query_lower for w in ["langue", "language", "parle", "officielle"])

    parts = []

    if wants_capital or wants_pop or (not wants_currency and not wants_language):
        parts.append(
            "Porto-Novo est la capitale officielle du Bénin. "
            "Cotonou est la capitale économique et le siège du gouvernement. "
            "Le Bénin compte environ 13 millions d'habitants (2024)."
        )

    if wants_currency:
        parts.append("Le Bénin utilise le Franc CFA (XOF), partagé avec 7 autres pays d'Afrique de l'Ouest.")

    if wants_language:
        parts.append("La langue officielle est le français. Le fon, le yoruba et le bariba sont très parlés.")

    if parts:
        return " ".join(parts)

    return (
        "Le Bénin est un pays d'Afrique de l'Ouest. "
        "Capitale : Porto-Novo. Population : ~13 millions. "
        "Monnaie : Franc CFA. Langue officielle : français."
    )


@tool(description="Effectue un calcul mathématique : addition, soustraction, multiplication, division, puissance")
def calculator(expression: str) -> str:
    """Évalue une expression mathématique de façon sécurisée."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Expression non autorisée : {expression}"
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {e}"


# ─────────────────────────────────────────────
# 2. Le client LLM — HTTP pur, zéro requests
# ─────────────────────────────────────────────

from zeroagent.core.llm import LLMClient

llm = LLMClient(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
    api_key=os.environ.get("LLM_API_KEY", ""),
    max_tokens=512,
    temperature=0.1,   # bas pour des réponses plus déterministes
)


# ─────────────────────────────────────────────
# 3. L'agent — LLM + Harness
# ─────────────────────────────────────────────

from zeroagent.harness.agent import Agent

SYSTEM_PROMPT = """Tu es un assistant concis. Tu réponds en français.

Quand tu as suffisamment d'informations pour répondre, tu réponds IMMÉDIATEMENT
avec ce format exact :

Final Answer: [ta réponse complète ici]

Ne fais pas plus d'un ou deux appels d'outils. Dès que tu as la réponse, utilise Final Answer."""

agent = Agent(
    llm=llm,
    tools=[search, calculator],
    system_prompt=SYSTEM_PROMPT,
    verbose=True,
    max_iterations=5,
)


# ─────────────────────────────────────────────
# 4. La démo
# ─────────────────────────────────────────────

QUESTIONS = [
    "Quelle est la capitale du Bénin et combien d'habitants a ce pays ?",
    "Combien font 1337 multiplié par 42 ?",
]

def separator(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)

def run_demo() -> None:
    print("\n" + "═" * 60)
    print("  zeroagent — Day 1 Demo")
    print("  Agent IA · Python pur · Zéro dépendance")
    print("═" * 60)
    print(f"\n  Tools    : {agent.tools}")
    print(f"  Modèle   : {llm.model}")
    print(f"  Provider : {llm._provider}")

    for i, question in enumerate(QUESTIONS, 1):
        separator(f"Question {i}")
        print(f"\n  → {question}\n")
        try:
            result = agent.run_full(question)
            separator("Réponse finale")
            print(f"\n  {result.answer}")
            print(
                f"\n  [ {result.iterations} itération(s) · "
                f"{result.tool_calls_made} tool call(s) · "
                f"stop: {result.stop_reason.value} ]"
            )
            agent.reset()
        except Exception as e:
            print(f"\n  Erreur : {e}")
            sys.exit(1)

    separator()
    print("\n  Démo terminée. Repo : github.com/Tahsine/zeroagent\n")


if __name__ == "__main__":
    run_demo()
