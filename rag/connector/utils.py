from functools import lru_cache

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from comps import CustomLogger
from rag.connector.embedding.hashable_huggingface_endpoint import HashableHuggingFaceEndpointEmbeddings
from rag.connector.embedding.mosec_embeddings import MosecEmbeddings
from rag.connector.vectorstore.base import VectorStore
from rag.connector.vectorstore import MilvusVectorStore
from langchain_core.embeddings import Embeddings

logger = CustomLogger("rag_connector_utils")

@lru_cache
def get_embedding_model(embedding_type, mosec_embedding_model, mosec_embedding_endpoint, tei_embedding_endpoint, local_embedding_model) -> Embeddings:
    """Create the embedding model."""
    if embedding_type == "MOSEC":
        return MosecEmbeddings(model=mosec_embedding_model)
    elif embedding_type == "TEI":
        return HashableHuggingFaceEndpointEmbeddings(model=tei_embedding_endpoint)
    elif embedding_type == "LOCAL":
        if any([key_word in local_embedding_model for key_word in ["bge"]]):
            return HuggingFaceBgeEmbeddings(model_name=local_embedding_model)
        else:
            return HuggingFaceEmbeddings(model_name=local_embedding_model)
    else:
        raise RuntimeError("Unable to find any supported embedding model.")

@lru_cache
def get_vectorstore(knowledge_name,
                    vs_type,
                    embedding_model
                    ) -> VectorStore:
    """Get the vectorstore"""
    vectorstore = None
    logger.info(f"Using {vs_type} as db to create vectorstore")
    if vs_type == "milvus":
        vectorstore = MilvusVectorStore(embedding_model=embedding_model, collection_name=knowledge_name)
    else:
        raise ValueError(f"{vs_type} vector database is not supported")
    logger.info("Vector store created")
    return vectorstore
