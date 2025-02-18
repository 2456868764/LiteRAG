import uuid
from typing import List
from langchain_core.documents import Document
from rag.module.indexing.splitter.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter


def split_smaller_chunks(documents: List[Document], smaller_chunk_size: int):
    doc_ids = [doc.metadata['id'] for doc in documents]
    tot_docs = []

    # child_splitter = RecursiveCharacterTextSplitter(chunk_size=smaller_chunk_size,
    #                                                 chunk_overlap=0)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=smaller_chunk_size,
                                                    chunk_overlap=0)
    for i, doc in enumerate(documents):
        parent_id = doc_ids[i]
        sub_docs = child_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata['parent_id'] = parent_id
            sub_doc.metadata['id'] = str(uuid.uuid4())
            sub_doc.metadata['multi_vector_type'] = "text small-to-big"
            tot_docs.append(sub_doc)
    return tot_docs


