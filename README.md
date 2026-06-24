# zeroagent

**Agent IA + Workflow SDK en Python pur. Zéro dépendance externe.**

![zero dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![version](https://img.shields.io/badge/version-0.1.0-blue)
![tests](https://img.shields.io/badge/tests-139%20passing-brightgreen)
![status](https://img.shields.io/badge/status-in%20development-orange)

---

## Ce que c'est

`zeroagent` est un SDK minimaliste pour construire des agents IA et des workflows en Python,
sans installer la moindre librairie externe. Tout repose sur la stdlib Python.

**Pourquoi ?**

- `pip install langchain` tire 80+ dépendances transitives. Pour beaucoup de cas d'usage, c'est inutile.
- Comprendre ce qui se passe réellement dans une boucle agent est impossible quand tout est abstrait derrière des classes opaques.
- Les environnements contraints (Kaggle, containers Alpine, Raspberry Pi) ne peuvent pas se permettre ce poids.

`zeroagent` expose les mêmes primitives — agent ReAct, tool registry, memory, workflow DAG —
dans un code que tu peux lire en une après-midi.

**Fonctionnel aujourd'hui :** CLI one-shot, chat interactif, streaming token-par-token,
tool calling, boucle ReAct complète, mémoire conversationnelle (Buffer, Window, Summary),
API SDK publique stable.

---

## Quick start

Aucune dépendance à installer pour zeroagent lui-même :

```bash
# 1. Cloner
git clone https://github.com/Tahsine/zeroagent.git
cd zeroagent

# 2. Lancer directement (zéro pip install)
python -m zeroagent "Calcule 123 * 456"

# 3. Mode chat interactif
python -m zeroagent --chat

# 4. Avec streaming token-par-token
python -m zeroagent --chat --stream

# 5. Voir la boucle ReAct (thought → action → observation)
python -m zeroagent "Calcule (15 + 27) * 3" --verbose
```

### Configuration

Copier ou créer un fichier `.env` à la racine :

```env
LLM_BASE_URL=https://ollama.com/v1
LLM_MODEL=minimax-m3:cloud
LLM_API_KEY=ta_clé_api
```

`zeroagent` fonctionne avec **tout provider OpenAI-compatible** : OpenAI, Ollama (local ou cloud),
Kaggle LLM Server, Anthropic, etc.

---

## API SDK

```python
from zeroagent import Agent, tool, LLMClient

@tool(description="Recherche des informations")
def search(query: str) -> str:
    return f"Résultats pour : {query}"

llm = LLMClient(base_url="https://api.openai.com/v1", api_key="sk-...")
agent = Agent(llm=llm, tools=[search])

result = agent.run("Quelle est la capitale du Bénin ?")
print(result)
```

### Imports disponibles

```python
# Entrées principales
from zeroagent import Agent, tool, LLMClient

# Types LLM
from zeroagent import Message, LLMResponse, ToolCall

# Registry
from zeroagent import ToolRegistry, ToolSchema

# Mémoire
from zeroagent import BufferMemory, WindowMemory, SummaryMemory

# Résultats et contrôle de boucle
from zeroagent import RunResult, StopReason, RunConfig

# Avancé (hooks, introspection)
from zeroagent import ParsedAction, ActionType, ExecutionResult
```

### Mémoire conversationnelle

```python
from zeroagent import Agent, LLMClient, BufferMemory, WindowMemory, SummaryMemory

# Buffer complet — garde toute la conversation
agent = Agent(llm=llm, memory=BufferMemory())

# Fenêtre glissante — garde les 10 derniers messages
agent = Agent(llm=llm, memory=WindowMemory(k=10))

# Résumé automatique — compresse les vieux messages via LLM
agent = Agent(llm=llm, memory=SummaryMemory(llm=llm, max_messages=20, keep_recent=6))
```

### Hooks (observe le comportement interne de l'agent)

```python
agent = Agent(
    llm=llm,
    tools=[search],
    on_thought=lambda thought, i: print(f"[{i}] Thought: {thought}"),
    on_action=lambda name, args, i: print(f"[{i}] Action: {name}({args})"),
    on_observation=lambda name, obs, ok, i: print(f"[{i}] Obs: {obs}"),
    on_final=lambda answer, reason: print(f"Final ({reason}): {answer}"),
)
```

### run() vs run_full()

```python
# run() — retourne juste la réponse finale (str)
answer = agent.run("Quelle est la capitale du Bénin ?")

# run_full() — retourne le RunResult complet
result = agent.run_full("Quelle est la capitale du Bénin ?")
print(result.answer)           # str
print(result.stop_reason)      # StopReason.FINAL_ANSWER | MAX_ITERATIONS | ...
print(result.iterations)       # int
print(result.tool_calls_made)  # int
print(result.thoughts)         # list[str]
```

---

## CLI flags

| Flag | Défaut | Description |
|------|--------|-------------|
| `question` | — | Question one-shot (positionnel) |
| `--base-url` | `$LLM_BASE_URL` | URL du provider LLM |
| `--model` | `$LLM_MODEL` | Nom du modèle |
| `--api-key` | `$LLM_API_KEY` | Clé API |
| `--system-prompt` | *défaut* | Prompt système personnalisé |
| `--max-tokens` | `1024` | Limite de tokens générés |
| `--temperature` | `0.3` | Température d'échantillonnage |
| `--max-iterations` | `6` | Nombre max d'itérations ReAct |
| `--no-tools` | — | Désactive les outils par défaut |
| `--verbose` | — | Affiche Thought/Action/Observation |
| `--quiet` | — | Sortie minimale (juste la réponse) |
| `--fail-on-error` | — | Exit code 1 si le run échoue |
| `--chat` | — | Mode chat interactif (REPL) |
| `--stream` | — | Streaming token-par-token (chat uniquement) |
| `--env` | `.env` | Chemin custom vers un fichier .env |

---

## Architecture

```
src/zeroagent/
├── __init__.py       # Package root — re-exporte toute l'API publique
├── __main__.py       # CLI — argparse, couleurs, modes one-shot/chat/stream
├── sdk.py            # Point d'entrée public — tous les imports stables
├── demo.py           # Démo rapide
│
├── core/             # Fondations
│   ├── llm.py        # LLMClient — HTTP pur, OpenAI-compatible + Anthropic
│   ├── tools.py      # @tool décorateur + ToolRegistry (génère JSON Schema)
│   └── memory.py     # BufferMemory + WindowMemory
│
├── harness/          # Boucle ReAct complète
│   ├── agent.py      # Agent (façade publique : LLM + Registry + Memory + Loop)
│   ├── loop.py       # Boucle ReAct : Thought → Action → Observation
│   ├── parser.py     # Parse les décisions du LLM (tool call, final answer, thought)
│   ├── executor.py   # Exécute les actions, capture les observations
│   └── memory.py     # SummaryMemory (résumé automatique via LLM)
│
├── workflow/         # (en construction) DAG multi-nœuds
│
└── tests/            # 139 tests unitaires (stdlib unittest)
    ├── test_llm.py
    ├── test_tools.py
    ├── test_harness.py
    └── test_memory.py
```

---

## Test comparatif : zeroagent vs LangChain

Un script CLI **strictement identique** à zeroagent a été écrit avec LangChain
pour permettre une comparaison **côte à côte** en vidéo/démo :

```bash
# 1. Installer les dépendances LangChain
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-openai

# 2. Lancer le même test
python langchain-sdk-test.py "Calcule 123 * 456"
python langchain-sdk-test.py --chat --verbose
```

Mêmes flags, mêmes couleurs, même outil `calculator`, même `.env`.
Les différences visibles :
- **zeroagent** : zéro dépendance, code lisible et maîtrisé, protocole ReAct textuel
- **LangChain** : nécessite 40+ packages, boîte noire, mais tool calling natif structuré

Le fichier `langchain-sdk-test.py` est un outil de démonstration, pas une fonctionnalité du SDK.

---

## Ce que ce n'est pas

- Un framework de prod avec checkpointing, observabilité et distributed execution → use LangGraph
- Un wrapper autour d'une API spécifique → zeroagent est provider-agnostic
- Une réimplémentation de LangChain → les primitives sont similaires, les choix de design sont différents

---

## Roadmap

- [x] LLMClient HTTP pur (OpenAI-compatible + Anthropic natif)
- [x] `@tool` décorateur + ToolRegistry (JSON Schema auto-généré)
- [x] Boucle ReAct complète (Thought → Action → Observation)
- [x] Mémoire conversationnelle (Buffer, Window, Summary)
- [x] CLI one-shot, chat, streaming
- [x] API SDK publique (`from zeroagent import Agent, tool, LLMClient`)
- [ ] `arun()` — support async natif
- [ ] `workflow/` — DAG multi-nœuds (Node, State, Graph, parallel execution)
- [ ] Publication PyPI

---

## Statut

Développement en cours. Suivi public sur [LinkedIn](https://www.linkedin.com/in/byborrelle)
et [YouTube](https://youtube.com/@BorrelleDev).

Chaque phase est documentée publiquement dans `DESIGN.md`.

---

## Licence

MIT
