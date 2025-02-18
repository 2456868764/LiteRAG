from typing import Any

from langchain_huggingface import HuggingFaceEndpointEmbeddings


class HashableHuggingFaceEndpointEmbeddings(HuggingFaceEndpointEmbeddings):
    """A hashable version of HuggingFaceEndpointEmbeddings."""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, HashableHuggingFaceEndpointEmbeddings):
            return False
        return (
                self.model == other.model
                and self.task == other.task
                and self.huggingfacehub_api_token == other.huggingfacehub_api_token
        )

    def __hash__(self) -> int:
        return hash((self.model, self.task, self.huggingfacehub_api_token))
