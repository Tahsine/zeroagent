"""
workflow/graph.py — DAG de nœuds avec exécution sync + async.

Graph orchestre l'exécution d'un ensemble de Nodes connectés par des edges.
Supporte :
  - Edges simples : A → B
  - Edges conditionnels : A → fn(state) → B ou C
  - Exécution parallèle : nœuds sans dépendances non satisfaites via asyncio.gather()
  - Visualisation ASCII du DAG
  - Exécution sync (graph.run()) et async (graph.arun())

Design :
  - Le Graph ne modifie pas le State directement — les nœuds le font
  - L'état est passé par copie entre nœuds (State.copy()) pour éviter
    les mutations silencieuses — sauf pour les nœuds parallèles qui
    partagent une copie commune en lecture et mergent leurs résultats
  - Détection de cycle à l'ajout d'edge (pas à l'exécution)
  - Pas de checkpointing — hors scope, responsabilité de l'utilisateur

Terminologie :
  - Entry node  : nœud de départ (set_entry)
  - Finish node : nœud(s) de fin (set_finish) — plusieurs autorisés
  - Edge        : connexion entre nœuds
  - Conditional edge : edge avec une fonction de routage fn(state) -> str

Usage :
    graph = Graph()
    graph.add_node(Node("fetch", fn=fetch_fn))
    graph.add_node(Node("analyze", fn=analyze_fn))
    graph.add_edge("fetch", "analyze")
    graph.set_entry("fetch")
    graph.set_finish("analyze")

    result_state = graph.run(State({"query": "test"}))

    # Conditional edge
    def router(state: State) -> str:
        return "success" if state.get("ok") else "fallback"

    graph.add_conditional_edge("validate", router, {
        "success": "process",
        "fallback": "error_handler",
    })
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from zeroagent.workflow.node import Node
from zeroagent.workflow.state import State


# ---------------------------------------------------------------------------
# Types d'edges
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """Edge simple : source → target."""
    source: str
    target: str

    def __repr__(self) -> str:
        return f"{self.source} → {self.target}"


@dataclass
class ConditionalEdge:
    """
    Edge conditionnel : source → fn(state) → target.
    fn retourne une clé string. mapping résout la clé vers un nom de nœud.
    """
    source: str
    fn: Callable[[State], str]
    mapping: dict[str, str]   # {"key": "node_name"}

    def resolve(self, state: State) -> str:
        """Retourne le nom du nœud cible selon l'état."""
        key = self.fn(state)
        if key not in self.mapping:
            available = list(self.mapping.keys())
            raise ValueError(
                f"ConditionalEdge depuis '{self.source}': "
                f"clé '{key}' inconnue. Disponibles: {available}"
            )
        return self.mapping[key]

    def __repr__(self) -> str:
        targets = list(self.mapping.values())
        return f"{self.source} →? {targets}"


# ---------------------------------------------------------------------------
# GraphResult
# ---------------------------------------------------------------------------

@dataclass
class GraphResult:
    """Résultat d'une exécution de Graph."""
    state: State
    executed: list[str] = field(default_factory=list)   # nœuds exécutés dans l'ordre
    success: bool = True
    error: str | None = None

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"GraphResult({status} nodes={self.executed})"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Graph:
    """
    DAG de nœuds avec exécution sync + async.

    Usage minimal :
        graph = Graph()
        graph.add_node(Node("a", fn=fn_a))
        graph.add_node(Node("b", fn=fn_b))
        graph.add_edge("a", "b")
        graph.set_entry("a")
        graph.set_finish("b")
        result = graph.run(State({"x": 1}))

    Avec edges conditionnels :
        graph.add_conditional_edge("a", router_fn, {"ok": "b", "err": "c"})

    Exécution async :
        result = await graph.arun(State({"x": 1}))
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdge] = []
        self._entry: str | None = None
        self._finish: set[str] = set()

    # ------------------------------------------------------------------
    # Construction du graphe
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> "Graph":
        """Ajoute un nœud. Retourne self pour chaining."""
        if not isinstance(node, Node):
            raise TypeError(f"add_node attend un Node, got {type(node)}")
        if node.name in self._nodes:
            raise ValueError(f"Nœud '{node.name}' déjà enregistré")
        self._nodes[node.name] = node
        return self

    def add_edge(self, source: str, target: str) -> "Graph":
        """
        Ajoute un edge simple source → target.
        Vérifie l'absence de cycle après ajout.
        Retourne self pour chaining.
        """
        self._check_nodes_exist(source, target)
        self._edges.append(Edge(source=source, target=target))
        if self._has_cycle():
            self._edges.pop()
            raise ValueError(
                f"L'edge '{source}' → '{target}' crée un cycle"
            )
        return self

    def add_conditional_edge(
        self,
        source: str,
        fn: Callable[[State], str],
        mapping: dict[str, str],
    ) -> "Graph":
        """
        Ajoute un edge conditionnel.
        fn(state) retourne une clé, mapping résout vers un nœud cible.

        Exemple :
            graph.add_conditional_edge("validate", router, {
                "ok": "process",
                "error": "handler",
            })
        """
        self._check_node_exists(source)
        for key, target in mapping.items():
            self._check_node_exists(target)
        if not callable(fn):
            raise TypeError("fn doit être callable")
        self._conditional_edges.append(
            ConditionalEdge(source=source, fn=fn, mapping=mapping)
        )
        return self

    def set_entry(self, node_name: str) -> "Graph":
        """Définit le nœud d'entrée du graphe."""
        self._check_node_exists(node_name)
        self._entry = node_name
        return self

    def set_finish(self, *node_names: str) -> "Graph":
        """
        Définit un ou plusieurs nœuds de fin.
        L'exécution s'arrête quand un finish node est atteint.
        """
        for name in node_names:
            self._check_node_exists(name)
            self._finish.add(name)
        return self

    # ------------------------------------------------------------------
    # Exécution sync
    # ------------------------------------------------------------------

    def run(self, state: State | None = None) -> GraphResult:
        """
        Exécute le graphe de façon synchrone.
        Retourne un GraphResult avec l'état final et la liste des nœuds exécutés.
        """
        self._validate_graph()
        state = state or State()
        executed: list[str] = []
        current = self._entry

        while current is not None:
            node = self._nodes[current]
            try:
                state = node.run(state)
            except Exception as e:
                return GraphResult(
                    state=state,
                    executed=executed,
                    success=False,
                    error=f"Erreur dans '{current}': {e}",
                )
            executed.append(current)

            if current in self._finish:
                break

            current = self._next_node(current, state)

        return GraphResult(state=state, executed=executed, success=True)

    # ------------------------------------------------------------------
    # Exécution async
    # ------------------------------------------------------------------

    async def arun(self, state: State | None = None) -> GraphResult:
        """
        Exécute le graphe de façon asynchrone.
        Les nœuds parallèles (même source, pas de dépendances entre eux)
        sont exécutés via asyncio.gather().
        """
        self._validate_graph()
        state = state or State()
        executed: list[str] = []
        current = self._entry

        while current is not None:
            # Chercher les nœuds parallèles à ce niveau
            parallel = self._parallel_targets(current)

            if parallel:
                # Exécuter le nœud courant + ses parallèles en gather()
                node = self._nodes[current]
                try:
                    state, parallel_executed = await self._run_parallel(
                        current, parallel, state
                    )
                    executed.extend(parallel_executed)
                except Exception as e:
                    return GraphResult(
                        state=state,
                        executed=executed,
                        success=False,
                        error=f"Erreur parallèle depuis '{current}': {e}",
                    )
                # Après parallèles, chercher le prochain nœud commun
                current = self._next_after_parallel(parallel, state)
            else:
                node = self._nodes[current]
                try:
                    state = await node.arun(state)
                except Exception as e:
                    return GraphResult(
                        state=state,
                        executed=executed,
                        success=False,
                        error=f"Erreur dans '{current}': {e}",
                    )
                executed.append(current)

                if current in self._finish:
                    break

                current = self._next_node(current, state)

        return GraphResult(state=state, executed=executed, success=True)

    # ------------------------------------------------------------------
    # Visualisation ASCII
    # ------------------------------------------------------------------

    def visualize(self) -> str:
        """
        Retourne une représentation ASCII du DAG.

        Exemple :
            ┌─────────┐
            │  fetch  │ (entry)
            └────┬────┘
                 │
            ┌────▼────┐
            │ analyze │
            └────┬────┘
                 │
            ┌────▼────┐
            │ output  │ (finish)
            └─────────┘
        """
        if not self._entry:
            return f"Graph '{self.name}' — vide (pas d'entry)"

        lines: list[str] = [f"Graph '{self.name}'", ""]
        visited: set[str] = set()
        self._ascii_node(self._entry, lines, visited, depth=0)
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = len(self._edges) + len(self._conditional_edges)
        entry = self._entry or "?"
        return f"Graph(name={self.name!r}, nodes={n}, edges={e}, entry={entry!r})"

    # ------------------------------------------------------------------
    # Interne — navigation
    # ------------------------------------------------------------------

    def _next_node(self, current: str, state: State) -> str | None:
        """Retourne le prochain nœud à exécuter après `current`."""
        # Edges conditionnels en priorité
        for ce in self._conditional_edges:
            if ce.source == current:
                return ce.resolve(state)

        # Edge simple
        for edge in self._edges:
            if edge.source == current:
                return edge.target

        return None

    def _parallel_targets(self, current: str) -> list[str]:
        """
        Retourne les nœuds qui peuvent être exécutés en parallèle
        depuis `current` — c'est-à-dire les nœuds qui ont plusieurs
        edges entrants dont tous les prédécesseurs ont déjà été exécutés,
        OU les nœuds qui partagent un prédécesseur commun sans dépendance
        entre eux.

        Pour l'instant on utilise un critère simple :
        si un nœud a plusieurs edges sortants vers des nœuds qui n'ont
        aucune dépendance entre eux → parallèles.

        Note : les conditional edges ne sont PAS parallélisés
        (on ne peut pas savoir à l'avance quelle branche sera choisie).
        """
        # Chercher les edges simples depuis current
        targets = [e.target for e in self._edges if e.source == current]

        # Si 0 ou 1 target → pas de parallélisme ici
        if len(targets) <= 1:
            return []

        # Vérifier qu'aucun de ces targets n'est prédécesseur d'un autre
        # (sinon il y a une dépendance et on ne peut pas paralléliser)
        target_set = set(targets)
        for t in targets:
            successors = self._all_successors(t)
            if successors & target_set - {t}:
                # t est prédécesseur d'un autre target → pas parallélisable
                return []

        return targets

    def _all_successors(self, node_name: str) -> set[str]:
        """Retourne tous les successeurs transitifs d'un nœud."""
        visited: set[str] = set()
        queue = [node_name]
        while queue:
            n = queue.pop()
            for edge in self._edges:
                if edge.source == n and edge.target not in visited:
                    visited.add(edge.target)
                    queue.append(edge.target)
        return visited

    async def _run_parallel(
        self,
        source: str,
        targets: list[str],
        state: State,
    ) -> tuple[State, list[str]]:
        """
        Exécute les nœuds `targets` en parallèle via asyncio.gather().
        Chaque nœud reçoit une copie du state.
        Les résultats sont mergés dans le state final dans l'ordre des targets.
        """
        async def run_one(node_name: str) -> State:
            node = self._nodes[node_name]
            state_copy = state.copy()
            return await node.arun(state_copy)

        results = await asyncio.gather(*[run_one(t) for t in targets])

        # Merger tous les résultats dans le state courant
        for result_state in results:
            state.update(result_state.to_dict())

        executed = list(targets)
        return state, executed

    def _next_after_parallel(
        self,
        parallel_nodes: list[str],
        state: State,
    ) -> str | None:
        """
        Après une exécution parallèle, trouve le nœud commun suivant.
        Le nœud commun est celui qui a des edges entrants depuis tous les
        nœuds parallèles.
        """
        if not parallel_nodes:
            return None

        # Successeurs de chaque nœud parallèle
        successors_per_node = [
            {e.target for e in self._edges if e.source == n}
            for n in parallel_nodes
        ]

        if not successors_per_node:
            return None

        # Intersection : nœuds accessibles depuis tous les parallèles
        common = successors_per_node[0]
        for s in successors_per_node[1:]:
            common = common & s

        if not common:
            return None

        # Retourner le premier commun (ordre d'ajout des edges)
        for edge in self._edges:
            if edge.target in common:
                return edge.target

        return None

    # ------------------------------------------------------------------
    # Interne — validation + cycles
    # ------------------------------------------------------------------

    def _validate_graph(self) -> None:
        """Vérifie que le graphe est exécutable."""
        if not self._entry:
            raise RuntimeError("Graph sans entry node. Appelle set_entry() d'abord.")
        if self._entry not in self._nodes:
            raise RuntimeError(f"Entry node '{self._entry}' non enregistré.")
        if not self._finish:
            raise RuntimeError("Graph sans finish node. Appelle set_finish() d'abord.")

    def _has_cycle(self) -> bool:
        """Détection de cycle via DFS (coloration tri-état)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for edge in self._edges:
                if edge.source == node:
                    if color.get(edge.target, WHITE) == GRAY:
                        return True
                    if color.get(edge.target, WHITE) == WHITE:
                        if dfs(edge.target):
                            return True
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def _check_node_exists(self, name: str) -> None:
        if name not in self._nodes:
            raise ValueError(f"Nœud '{name}' non enregistré dans le Graph")

    def _check_nodes_exist(self, *names: str) -> None:
        for name in names:
            self._check_node_exists(name)

    # ------------------------------------------------------------------
    # Interne — visualisation ASCII
    # ------------------------------------------------------------------

    def _ascii_node(
        self,
        name: str,
        lines: list[str],
        visited: set[str],
        depth: int,
    ) -> None:
        if name in visited:
            lines.append(f"{'  ' * depth}↩ {name} (déjà affiché)")
            return
        visited.add(name)

        tags = []
        if name == self._entry:
            tags.append("entry")
        if name in self._finish:
            tags.append("finish")
        tag_str = f" ({', '.join(tags)})" if tags else ""

        indent = "  " * depth
        width = max(len(name) + 4, 10)
        bar = "─" * (width - 2)

        lines.append(f"{indent}┌{bar}┐")
        lines.append(f"{indent}│ {name.center(width - 4)} │{tag_str}")
        lines.append(f"{indent}└{bar}┘")

        # Edges conditionnels
        for ce in self._conditional_edges:
            if ce.source == name:
                lines.append(f"{indent}  │ (conditionnel)")
                for key, target in ce.mapping.items():
                    lines.append(f"{indent}  ├─[{key}]─▶")
                    self._ascii_node(target, lines, visited, depth + 2)

        # Edges simples (parallèles ou séquentiels)
        targets = [e.target for e in self._edges if e.source == name]
        if len(targets) > 1:
            lines.append(f"{indent}  │ (parallèle)")
            for target in targets:
                lines.append(f"{indent}  ├─▶")
                self._ascii_node(target, lines, visited, depth + 2)
        elif len(targets) == 1:
            lines.append(f"{indent}  │")
            lines.append(f"{indent}  ▼")
            self._ascii_node(targets[0], lines, visited, depth)
