# zeroagent

**Agent IA + Workflow SDK en Python pur. Zéro dépendance externe.**

![zero dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
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
tool calling, boucle ReAct complète.

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
├── __main__.py       # CLI — argparse, couleurs, modes one-shot/chat/stream
├── sdk.py            # Point d'entrée public (imports à venir)
├── demo.py           # Démo rapide
│
├── core/             # Fondations
│   ├── llm.py        # LLMClient — HTTP pur, OpenAI-compatible + Anthropic
│   ├── tools.py      # @tool décorateur + ToolRegistry (génère JSON Schema)
│   └── memory.py     # BufferMemory conversationnelle
│
├── harness/          # Boucle ReAct complète
│   ├── agent.py      # Agent (façade publique : LLM + Registry + Memory + Loop)
│   ├── loop.py       # Boucle ReAct : Thought → Action → Observation
│   ├── parser.py     # Parse les décisions du LLM (tool call, final answer, thought)
│   └── executor.py   # Exécute les actions, capture les observations
│
├── workflow/         # (en construction) DAG multi-nœuds
│
└── tests/            # Tests unitaires (stdlib unittest)
    ├── test_llm.py
    ├── test_tools.py
    ├── test_harness.py
    └── test_memory.py
```

---

## API SDK

À venir. L'import public final sera :

```python
from zeroagent import Agent, tool
from zeroagent.core.llm import LLMClient

@tool(description="Recherche des informations")
def search(query: str) -> str:
    return f"Résultats pour : {query}"

llm = LLMClient(base_url="https://api.openai.com/v1", api_key="sk-...")
agent = Agent(llm=llm, tools=[search])

result = agent.run("Quelle est la capitale du Bénin ?")
print(result)
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

## Statut

Développement en cours. Suivi public sur [LinkedIn](https://linkedin.com/in/ton-profil)
et [YouTube](https://youtube.com/@BorrelleDev).

Chaque phase est documentée publiquement dans `DESIGN.md`.

---

## Licence

MIT
