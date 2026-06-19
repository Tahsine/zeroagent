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

---

## API cible (pas encore fonctionnel)

```python
from zeroagent import Agent, tool
from zeroagent.core.llm import LLMClient

# 1. Définir un outil
@tool(description="Recherche des informations sur le web")
def search(query: str) -> str:
    # ton implémentation
    return f"Résultats pour : {query}"

# 2. Créer un client LLM (compatible OpenAI)
llm = LLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-..."
)

# 3. Assembler l'agent (LLM + Harness)
agent = Agent(llm=llm, tools=[search])

# 4. Lancer
result = agent.run("Quelle est la capitale du Bénin ?")
print(result)
```

**Workflow multi-nœuds :**

```python
from zeroagent.workflow import Graph, Node, State

def fetch_data(state: State) -> State:
    state["data"] = "..."
    return state

def process(state: State) -> State:
    state["result"] = state["data"].upper()
    return state

graph = Graph()
graph.add_node(Node("fetch", fn=fetch_data))
graph.add_node(Node("process", fn=process))
graph.add_edge("fetch", "process")

final_state = graph.run(State())
```

---

## Architecture

```
zeroagent/
├── core/
│   ├── llm.py        # LLMClient — HTTP pur, OpenAI-compatible
│   ├── tools.py      # @tool décorateur + ToolRegistry
│   └── memory.py     # BufferMemory + SummaryMemory
│
├── harness/          # Ce qui transforme un LLM en agent
│   ├── parser.py     # Parse les décisions du LLM
│   ├── executor.py   # Exécute les actions, récupère observations
│   ├── loop.py       # Boucle ReAct (Thought → Action → Observation)
│   └── agent.py      # Agent = LLM + Harness (assemblage final)
│
├── workflow/
│   ├── node.py       # Unité de travail
│   ├── state.py      # État partagé entre nœuds
│   └── graph.py      # DAG — routing, exécution, cycles
│
├── sdk.py            # Point d'entrée public
└── __main__.py       # CLI : python -m zeroagent "..."
```

---

## Ce que ce n'est pas

- Un framework de prod avec checkpointing, observabilité, et distributed execution → use LangGraph
- Un wrapper autour d'une API spécifique → zeroagent est provider-agnostic
- Une réimplémentation de LangChain → les primitives sont similaires, les choix de design sont différents

---

## Statut

Développement en cours. Suivi public sur [LinkedIn](https://linkedin.com/in/ton-profil) et [YouTube](https://youtube.com/@BorrelleDev).

Chaque phase est documentée publiquement dans `DESIGN.md`.

---

## Licence

MIT
