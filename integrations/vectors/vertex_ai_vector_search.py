"""Vertex AI Vector Search backend.

Implements the VectorStore contract over Vertex Matching Engine using lazy
google-cloud-aiplatform imports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from integrations.vectors.base import Chunk, SearchHit, VectorStore


class VertexAiVectorSearchStore(VectorStore):
    provider = "vertex_ai_vector_search"

    def __init__(
        self,
        *,
        project: str,
        location: str,
        index_endpoint_id: str,
        deployed_index_id: str,
        index_id: Optional[str] = None,
    ) -> None:
        if not project:
            raise ValueError("VertexAiVectorSearchStore requires project")
        if not location:
            raise ValueError("VertexAiVectorSearchStore requires location")
        if not index_endpoint_id:
            raise ValueError("VertexAiVectorSearchStore requires index_endpoint_id")
        if not deployed_index_id:
            raise ValueError("VertexAiVectorSearchStore requires deployed_index_id")
        self.project = project
        self.location = location
        self.index_endpoint_id = index_endpoint_id
        self.deployed_index_id = deployed_index_id
        self.index_id = index_id
        self._index_client = None
        self._match_client = None
        self._dimension: Optional[int] = None

    def _clients(self):
        if self._index_client is not None and self._match_client is not None:
            return self._index_client, self._match_client
        try:
            from google.cloud import aiplatform_v1  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-cloud-aiplatform must be installed for vertex_ai_vector_search. "
                "Install: `pip install google-cloud-aiplatform`"
            ) from exc
        endpoint = f"{self.location}-aiplatform.googleapis.com"
        self._index_client = aiplatform_v1.IndexServiceClient(
            client_options={"api_endpoint": endpoint}
        )
        self._match_client = aiplatform_v1.MatchServiceClient(
            client_options={"api_endpoint": endpoint}
        )
        return self._index_client, self._match_client

    def _index_name(self) -> Optional[str]:
        if not self.index_id:
            return None
        return (
            f"projects/{self.project}/locations/{self.location}/indexes/{self.index_id}"
        )

    def _index_endpoint_name(self) -> str:
        return (
            f"projects/{self.project}/locations/{self.location}/indexEndpoints/"
            f"{self.index_endpoint_id}"
        )

    def ensure_index(self, *, dimension: int, metric: str = "cosine") -> None:
        del metric
        if self._dimension is not None and self._dimension != dimension:
            raise ValueError(
                f"Vertex Vector Search dimension mismatch: existing={self._dimension}, "
                f"requested={dimension}"
            )
        self._dimension = dimension
        index_client, _ = self._clients()
        # Validate endpoint and optional index exist; infra is provisioned via IaC.
        index_client.get_index_endpoint(name=self._index_endpoint_name())
        if self._index_name():
            index_client.get_index(name=self._index_name())

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        if self._index_name() is None:
            raise ValueError(
                "Vertex upsert requires `index_id` configured for stream updates."
            )
        index_client, _ = self._clients()
        if self._dimension is None:
            self._dimension = len(chunks[0].embedding)
        try:
            from google.cloud import aiplatform_v1  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-cloud-aiplatform must be installed for vertex_ai_vector_search."
            ) from exc

        datapoints = []
        for chunk in chunks:
            restrictions = [
                aiplatform_v1.IndexDatapoint.Restriction(
                    namespace="tenant_id", allow_list=[str(chunk.tenant_id)]
                ),
                aiplatform_v1.IndexDatapoint.Restriction(
                    namespace="integration_id", allow_list=[str(chunk.integration_id)]
                ),
            ]
            datapoints.append(
                aiplatform_v1.IndexDatapoint(
                    datapoint_id=chunk.chunk_id,
                    feature_vector=list(chunk.embedding),
                    restricts=restrictions,
                    crowding_tag=aiplatform_v1.IndexDatapoint.CrowdingTag(
                        crowding_attribute=str(chunk.doc_id)
                    ),
                )
            )
        index_client.upsert_datapoints(index=self._index_name(), datapoints=datapoints)
        return len(chunks)

    def query(
        self,
        *,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        del text
        if embedding is None:
            raise ValueError("VertexAiVectorSearchStore.query requires `embedding`")
        _, match_client = self._clients()
        try:
            from google.cloud import aiplatform_v1  # type: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-cloud-aiplatform must be installed for vertex_ai_vector_search."
            ) from exc

        namespace_filters = []
        for key, value in (filters or {}).items():
            if isinstance(value, (list, tuple, set)):
                allow = [str(v) for v in value]
            else:
                allow = [str(value)]
            namespace_filters.append(
                aiplatform_v1.IndexDatapoint.Restriction(
                    namespace=key, allow_list=allow
                )
            )

        query = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=aiplatform_v1.IndexDatapoint(
                datapoint_id="query",
                feature_vector=list(embedding),
                restricts=namespace_filters,
            ),
            neighbor_count=k,
        )
        resp = match_client.find_neighbors(
            request=aiplatform_v1.FindNeighborsRequest(
                index_endpoint=self._index_endpoint_name(),
                deployed_index_id=self.deployed_index_id,
                queries=[query],
                return_full_datapoint=False,
            )
        )
        out: List[SearchHit] = []
        for neighbors in resp.nearest_neighbors:
            for n in neighbors.neighbors:
                out.append(
                    SearchHit(
                        chunk_id=str(n.datapoint.datapoint_id),
                        doc_id="",
                        text="",
                        score=float(n.distance),
                        source_uri="",
                        metadata={},
                    )
                )
        return out[:k]

    def delete_by_doc(self, doc_id: str, *, tenant_id: Optional[str] = None) -> int:
        del doc_id, tenant_id
        # Matching Engine supports remove_datapoints by id; callers should keep
        # chunk id lists for exact deletion.
        return 0

    def stats(self) -> Dict[str, Any]:
        index_client, _ = self._clients()
        endpoint = index_client.get_index_endpoint(name=self._index_endpoint_name())
        return {
            "provider": self.provider,
            "project": self.project,
            "location": self.location,
            "index_endpoint_id": self.index_endpoint_id,
            "deployed_index_id": self.deployed_index_id,
            "index_id": self.index_id,
            "deployed_indexes": len(getattr(endpoint, "deployed_indexes", []) or []),
            "dimension": self._dimension,
        }
