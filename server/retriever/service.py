import json
import os
import time
from fastapi import HTTPException

from rag.common.configuration import config
from rag.connector.utils import get_embedding_model, get_vectorstore
from comps import (
    CustomLogger,
    EmbedDoc,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("retriever_milvus")

# get embedding model
embedding_model = get_embedding_model(embedding_type=config.embedding.embedding_type,
                                        mosec_embedding_model=config.embedding.mosec_embedding_model,
                                        mosec_embedding_endpoint=config.embedding.mosec_embedding_endpoint,
                                        tei_embedding_endpoint=config.embedding.tei_embedding_endpoint,
                                        local_embedding_model=config.embedding.local_embedding_model)


@register_microservice(
    name="opea_service@retriever_milvus",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@register_statistics(names=["opea_service@retriever_milvus"])
async def retrieve(input: EmbedDoc) -> SearchedDoc:
    logger.info(input)
    start = time.time()
    knowledge_name = input.knowledge_name
    if knowledge_name is None or knowledge_name.strip() == "":
        raise HTTPException(status_code=404, detail="knowledge name can't be empty")

    vs = get_vectorstore(knowledge_name=knowledge_name,
                         vs_type=config.vector_store.vector_store_type,
                         embedding_model=embedding_model)

    if input.search_type == "similarity":
        search_res = vs.search_docs_by_vector(input.embedding, input.k, None)
    elif input.search_type == "similarity_distance_threshold":
        if input.distance_threshold is None:
            raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
        search_res = vs.search_docs_by_vector(input.embedding, input.k, input.distance_threshold)
    elif input.search_type == "mmr":
        search_res = vs.search_docs_by_mmr(input.text, input.k, input.fetch_k, input.lambda_mult)

    searched_docs = []
    for r in search_res:
        searched_docs.append(TextDoc(text=r.page_content))
    result = SearchedDoc(retrieved_docs=searched_docs, initial_query=input.text)
    statistics_dict["opea_service@retriever_milvus"].append_latency(time.time() - start, None)
    logger.info(result)
    return result


if __name__ == "__main__":
    opea_microservices["opea_service@retriever_milvus"].start()