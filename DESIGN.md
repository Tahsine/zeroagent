# DESIGN.md — Décisions d'architecture

Ce document trace les décisions de design prises pendant la Phase 0,
avec le raisonnement derrière chaque choix. Il évolue au fil des phases.

---

## Philosophie générale

**Un agent IA = LLM + Harness.**

Le LLM génère du texte. Le harness lui donne les outils, interprète ses décisions,
exécute les actions, et boucle jusqu'à la réponse finale. Sans harness, c'est un
chatbot. Avec harness, c'est un agent.

`zeroagent` sépare explicitement ces deux responsabilités dans sa structure de code.

---

## Décision 1 — Zéro dépendance externe

**Choix :** stdlib Python uniquement. Pas de `requests`, `httpx`, `pydantic`, `tenacity`, rien.

**Pourquoi :**
- Instalable dans n'importe quel environnement sans résoudre de conflits de versions
- Le code reste lisible et auditable sans connaître des libs tierces
- `http.client` + `json` + `asyncio` couvrent 100% des besoins pour un LLM client

**Exceptions acceptées :**
- `numpy` uniquement si on implémente un vector store (Phase optionnelle).
  Dans ce cas, `numpy` sera une dépendance optionnelle déclarée dans `pyproject.toml`
  sous `[project.optional-dependencies]`.

---

## Décision 2 — Agent-first, Workflow ensuite

**Choix :** construire l'agent standalone (Phases 1-3) avant le workflow (Phase 4).

**Pourquoi :**
- Un agent fonctionne seul, peut être démontré seul, peut être publié seul
- Le workflow est une couche au-dessus : un nœud de workflow peut être un Agent
- Évite de coupler les deux avant de comprendre chacun en profondeur

**Conséquence :** `harness/agent.py` n'a aucune dépendance sur `workflow/`.
La dépendance va dans le sens inverse : `workflow/node.py` peut accepter un Agent
comme fonction de nœud.

---

## Décision 3 — Interface publique : classe directe

**Choix :** `Agent(llm=client, tools=[...])` plutôt que builder pattern ou décorateur.

**Pourquoi :**
- Pythonique et immédiatement compréhensible
- Facile à sous-classer pour des comportements custom
- Le builder pattern sera disponible plus tard comme sucre syntaxique

**API publique cible (stable dès la Phase 3) :**
```python
agent = Agent(llm=client, tools=[...], memory=BufferMemory(k=10))
result = agent.run("question")          # sync
result = await agent.arun("question")   # async
```

---

## Décision 4 — Sync-first, async en bonus

**Choix :** implémenter sync d'abord. `arun()` sera ajouté en Phase 3 comme wrapper asyncio.

**Pourquoi :**
- 90% des use cases n'ont pas besoin d'async
- Async contamine toute l'architecture si introduit trop tôt
- Pattern classique : `arun` wrape `run` dans un executor thread, ou on réimplémente
  proprement avec `asyncio` une fois que la logique sync est stabilisée

---

## Décision 5 — Boucle ReAct comme pattern agent

**Choix :** implémenter la boucle Thought → Action → Observation du paper ReAct (Yao et al. 2022).

**Pourquoi :**
- Fondement théorique solide, peer-reviewed
- Implémenté par tous les frameworks majeurs (LangChain, LangGraph, AutoGPT...)
- Simple à comprendre, simple à débugger

**Ce qu'on fait différemment de LangGraph :**
- Pas de Pregel / parallel message passing — trop complexe pour notre scope
- Pas de checkpointing built-in — l'état est en mémoire, persistance = responsabilité de l'utilisateur
- Pas de streaming partial state — on stream le texte LLM, pas les states intermédiaires
- Loop controller explicite dans `harness/loop.py` plutôt que caché dans le graph runner

---

## Décision 6 — Pattern Node / State / Graph pour le workflow

**Choix :** conserver le vocabulaire Node/State/Graph, implémentation from scratch.

**Pourquoi Node/State/Graph et pas autre chose :**
- Vocabulaire issu de la théorie des graphes, pas une invention LangGraph
- Déjà utilisé par Airflow, Prefect, Luigi — les devs le connaissent
- `Node` = unité de travail, `State` = données partagées, `Graph` = orchestration

**Différences avec LangGraph :**

| LangGraph | zeroagent |
|-----------|-----------|
| State via `TypedDict` + annotations magiques | `State` = classe explicite wrappant un dict |
| Channels pour la réduction de state | Pas de channels — merge explicite dans la fonction du nœud |
| Checkpointing SQLite/Redis built-in | Pas de checkpointing — scope hors SDK minimal |
| Nœuds héritent de classes LangChain | Nœuds = fonctions Python pures `fn(state) -> state` |
| Conditional edges via `.add_conditional_edges()` | Idem — on garde cette API, c'est bien pensé |

**Améliorations prévues en Phase 4 (fin de projet) :**
- State avec schema validation sans pydantic (via `__annotations__` + inspection)
- Parallel node execution via `asyncio.gather` pour les nœuds sans dépendances
- Visualisation ASCII du DAG dans le CLI

---

## Décision 7 — Nom : zeroagent

**Choix :** `zeroagent`

**Pourquoi :**
- Court, mémorable
- "zero" communique immédiatement la proposition de valeur (zéro dépendance)
- `pip install zeroagent` — sonne bien
- Disponible sur PyPI (à vérifier avant publication)

---

## Ce qu'on ne construira pas (scope explicite)

- Observabilité / tracing (LangSmith, Langfuse) → hors scope
- Multi-agent orchestration (communication inter-agents) → Phase future possible
- RAG / vector store intégré → optionnel, avec numpy comme seule dep
- Deployment / serving → hors scope
- GUI → hors scope

---

*Dernière mise à jour : Phase 0*
