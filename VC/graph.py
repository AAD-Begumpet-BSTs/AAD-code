# graph.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Iterable

Vertex = int
Edge = Tuple[Vertex, Vertex]


@dataclass
class Graph:
    """
    Simple undirected graph representation for Vertex Cover experiments.
    Vertices are 0..n-1.
    """
    n: int
    adj: Dict[Vertex, Set[Vertex]] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def __post_init__(self):
        if not self.adj:
            self.adj = {v: set() for v in range(self.n)}

    def add_edge(self, u: Vertex, v: Vertex) -> None:
        """
        Add an undirected edge (u, v). Self-loops are ignored.
        """
        if u == v:
            return  # ignore self-loops
        if u < 0 or v < 0 or u >= self.n or v >= self.n:
            raise ValueError(f"Vertex out of range: ({u}, {v}) for n={self.n}")

        self.adj[u].add(v)
        self.adj[v].add(u)

        # Store each edge once with u < v to avoid duplicates
        if u > v:
            u, v = v, u
        if (u, v) not in self.edges:
            self.edges.append((u, v))

    @classmethod
    def from_edge_list(cls, n: int, edge_list: Iterable[Tuple[int, int]]) -> "Graph":
        """
        Build a graph from a list of edges (u, v), vertices in [0, n-1].
        """
        G = cls(n)
        for (u, v) in edge_list:
            G.add_edge(u, v)
        return G

    @classmethod
    def from_file(cls, path: str, one_based: bool = False) -> "Graph":
        """
        Load an undirected graph from a simple text file.

        Format:
            First line: n m          # number of vertices and edges
            Next m lines: u v        # edge endpoints (0- or 1-based indices)

        Args:
            path: path to the file.
            one_based: if True, input vertices are 1..n and will be converted
                       to 0..n-1 internally.
        """
        with open(path, "r") as f:
            first = f.readline().strip()
            if not first:
                raise ValueError("Empty file")

            parts = first.split()
            if len(parts) != 2:
                raise ValueError("First line must be 'n m'")
            n, m = map(int, parts)
            G = cls(n)

            for i in range(m):
                line = f.readline()
                if not line:
                    raise ValueError(f"Expected {m} edges, got only {i}")
                u_str, v_str = line.strip().split()
                u, v = int(u_str), int(v_str)
                if one_based:
                    u -= 1
                    v -= 1
                G.add_edge(u, v)

        return G

    def number_of_vertices(self) -> int:
        return self.n

    def number_of_edges(self) -> int:
        return len(self.edges)
