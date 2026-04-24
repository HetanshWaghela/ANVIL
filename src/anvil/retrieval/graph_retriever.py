"""Graph-aware expansion over an initial list of retrieved elements."""

from __future__ import annotations

from anvil.knowledge.graph_store import GraphStore
from anvil.schemas.retrieval import RetrievedChunk


class GraphRetriever:
    """Expand a list of retrieved chunks with graph neighbors.

    Given a base ranking (e.g. fused BM25 + vector hits), this pulls in the
    1-hop or 2-hop neighborhood of each chunk. Neighbors inherit a slightly
    attenuated score so they rank below the seeds but above unrelated chunks.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        element_index: dict[str, RetrievedChunk],
        max_hops: int = 1,
        decay_per_hop: float = 0.5,
    ) -> None:
        """
        Args:
            graph_store: loaded GraphStore
            element_index: map from element_id → a RetrievedChunk-shaped object
                that provides the content/metadata for neighbors we may add.
            max_hops: how far to expand (1 or 2 recommended)
            decay_per_hop: multiplicative decay factor for each hop
        """
        self.graph_store = graph_store
        self.element_index = element_index
        self.max_hops = max_hops
        self.decay_per_hop = decay_per_hop

    def expand(self, seeds: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return seeds plus their graph neighbors, de-duplicated."""
        if not seeds:
            return []

        seed_ids = [c.element_id for c in seeds]
        expanded_ids = self.graph_store.expand(seed_ids, max_hops=self.max_hops)

        out_by_id: dict[str, RetrievedChunk] = {s.element_id: s for s in seeds}

        # Compute hop count via simple BFS from seeds
        hop_counts = self._bfs_hops(seed_ids, expanded_ids)

        # Max seed score used as the baseline for neighbor scores
        max_seed_score = max((s.score for s in seeds), default=0.0)

        for node_id in expanded_ids:
            if node_id in out_by_id:
                # Update graph_hops if we had no hop info before
                hops = hop_counts.get(node_id, 0)
                out_by_id[node_id].scores.graph_hops = min(
                    out_by_id[node_id].scores.graph_hops or hops, hops
                )
                continue
            base = self.element_index.get(node_id)
            if base is None:
                continue
            hops = hop_counts.get(node_id, 1)
            decayed_score = max_seed_score * (self.decay_per_hop**hops)
            chunk = RetrievedChunk(
                element_id=base.element_id,
                paragraph_ref=base.paragraph_ref,
                element_type=base.element_type,
                content=base.content,
                page_number=base.page_number,
                score=decayed_score,
                retrieval_source="graph",
            )
            chunk.scores.graph_hops = hops
            chunk.scores.fused = decayed_score
            out_by_id[node_id] = chunk

        return sorted(out_by_id.values(), key=lambda c: c.score, reverse=True)

    def _bfs_hops(self, seeds: list[str], horizon: set[str]) -> dict[str, int]:
        """Return hop-count from nearest seed for every node in `horizon`."""
        distances: dict[str, int] = {s: 0 for s in seeds}
        frontier: set[str] = set(seeds)
        for d in range(1, self.max_hops + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in self.graph_store.graph:
                    continue
                neighbors = set(self.graph_store.graph.successors(node)) | set(
                    self.graph_store.graph.predecessors(node)
                )
                for nb in neighbors:
                    if nb in distances:
                        continue
                    if nb not in horizon:
                        continue
                    distances[nb] = d
                    next_frontier.add(nb)
            frontier = next_frontier
            if not frontier:
                break
        return distances
