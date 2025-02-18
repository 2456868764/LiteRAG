import os
import urllib
from typing import List, Optional, Union
from fastapi import File, Form, HTTPException, UploadFile
from comps import CustomLogger, opea_microservices, register_microservice
from rag.common.utils import validate_knowledge_name, run_in_thread_pool
from rag.connector.database.base import DB
from rag.connector.database.service.knowledge_file_service import KnowledgeFileService
from rag.connector.database.service.knowledge_service import KnowledgeService
from rag.common.configuration import config
from rag.connector.database.service.url_queue_service import URLQueueService
from rag.connector.utils import get_embedding_model, get_vectorstore
from rag.module.indexing.indexing import Indexing
from rag.module.knowledge_file import get_file_path, KnowledgeFile, clear_kb_folder, delete_kb_folder
from rag.common.api import BaseResponse, ListResponse
from rag.tasks.url_crawler import URLCrawler
import sched
import time
import threading
from concurrent.futures import ThreadPoolExecutor



logger = CustomLogger("prepare_doc_milvus")

# init chunk size
CHUNK_SIZE = config.splitter.chunk_size
OVERLAP_SIZE = config.splitter.chunk_overlap
SMALLER_CHUNK_SIZE = config.splitter.smaller_chunk_size

# initiate db repository service
knowledge_service = KnowledgeService()
knowledge_file_service = KnowledgeFileService()
url_queue_service = URLQueueService()

# get embedding model
embedding_model = get_embedding_model(embedding_type=config.embedding.embedding_type,
                                        mosec_embedding_model=config.embedding.mosec_embedding_model,
                                        mosec_embedding_endpoint=config.embedding.mosec_embedding_endpoint,
                                        tei_embedding_endpoint=config.embedding.tei_embedding_endpoint,
                                        local_embedding_model=config.embedding.local_embedding_model)

# 创建全局的线程池
crawler_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crawler")

def run_crawler():
    """
    运行爬虫的后台任务
    """
    crawler = URLCrawler()
    while True:
        try:
            logger.info("执行爬虫任务...")
            ok = crawler.run()
            if not ok:
                logger.info("没有待处理的URL，等待5秒...")
                time.sleep(5)
        except Exception as e:
            logger.error(f"爬虫运行出错: {str(e)}", exc_info=True)
            time.sleep(30)  # 出错后等待30秒再重试

def start_background_tasks():
    """
    启动后台任务
    """
    logger.info("启动后台任务...")
    crawler_executor.submit(run_crawler)



@register_microservice(name="opea_service@prepare_doc_milvus", endpoint="/v1/knowledge/upload_docs", host="0.0.0.0", port=6010)
async def upload_docs(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    knowledge_name: str = Form(""),
    chunk_size: int = Form(CHUNK_SIZE),
    chunk_overlap: int = Form(OVERLAP_SIZE),
    smaller_chunk_size: int = Form(SMALLER_CHUNK_SIZE),
):
    logger.info(f"[ uploaded_files ] files:{files} to knowledge_name:{knowledge_name}")
    # step 1. check knowledge name
    if not validate_knowledge_name(knowledge_name):
        raise HTTPException(status_code=403, detail="knowledge name format is forbidden")

    if knowledge_name is None or knowledge_name.strip() == "":
        raise HTTPException(status_code=404, detail="knowledge name can't be empty")

    # step 2. check knowledge name is existed or not
    kb = knowledge_service.load_kb_from_db(knowledge_name)
    if kb is None:
        raise HTTPException(status_code=500, detail="knowledge name hasn't existed")

    failed_files = {}
    # file_names = list(docs.keys())
    file_names = []

    # step 3. save files
    for result in save_files_in_thread(files, knowledge_name=knowledge_name, override=True):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # step 4. update docs and indexing
    result = update_docs(
        knowledge_name=knowledge_name,
        file_names=file_names,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        smaller_chunk_size=smaller_chunk_size,
    )
    failed_files.update(result.data["failed_files"])

    return BaseResponse(status="success", msg="upload files and vector embedding done", data={"failed_files": failed_files})


@register_microservice(name="opea_service@prepare_doc_milvus", endpoint="/v1/knowledge/list", host="0.0.0.0", port=6010)
async def list_knowledge():
    logger.info(f"[ list ]  knowledge")
    return ListResponse(data=knowledge_service.list_kbs_from_db())


@register_microservice(name="opea_service@prepare_doc_milvus", endpoint="/v1/knowledge/create", host="0.0.0.0", port=6010)
async def create_knowledge(
        knowledge_name: str = Form(""),
        weburl: str = Form(""),
        scraping_level: int = Form(1),
        link_tags: str = Form(""),
):
    logger.info(f"[ create ] knowledge_name:{knowledge_name}, weburl:{weburl},sraping_level:{scraping_level},link_tags:{link_tags}")
    # step 1. check knowledge name
    if not validate_knowledge_name(knowledge_name):
        raise HTTPException(status_code=403, detail="knowledge name format is forbidden")

    if knowledge_name is None or knowledge_name.strip() == "":
        raise HTTPException(status_code=404, detail="knowledge name can't be empty")

    # step 2. check knowledge name is existed or not
    kb = knowledge_service.load_kb_from_db(knowledge_name)
    if kb is not None:
        raise HTTPException(status_code=500, detail="knowledge name has existed")

    # step 3. add knowledge name to db
    try:
        embed_type = config.embedding.embedding_type
        vector_store_type = config.vector_store.vector_store_type
        res = knowledge_service.add_kb_to_db(knowledge_name, "", vector_store_type, embed_type, weburl=weburl, scraping_level=scraping_level, link_tags=link_tags)
        if not res:
            raise HTTPException(status_code=500, detail="save knowledge name failed")
    except Exception as e:
        msg = f"create knowledge name failed： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)
        raise HTTPException(status_code=500, detail=msg)

    return BaseResponse(status="success", msg=f"add knowledge name success: {knowledge_name}")


@register_microservice(name="opea_service@prepare_doc_milvus", endpoint="/v1/knowledge/files", host="0.0.0.0", port=6010)
async def create_knowledge(
        knowledge_name: str = Form(""),
):
    logger.info(f"[ files ] knowledge_name:{knowledge_name}")
    # step 1. check knowledge name
    if not validate_knowledge_name(knowledge_name):
        raise HTTPException(status_code=403, detail="knowledge name format is forbidden")

    if knowledge_name is None or knowledge_name.strip() == "":
        raise HTTPException(status_code=404, detail="knowledge name can't be empty")

    kb = knowledge_service.load_kb_from_db(knowledge_name)
    if kb is None:
        raise HTTPException(status_code=500, detail="knowledge name has not existed")

    # step 3. add knowledge name to db
    try:
        return BaseResponse(status="success", msg="", data=knowledge_file_service.list_files_from_db(knowledge_name))
    except Exception as e:
        msg = f"create knowledge name failed： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)
        raise HTTPException(status_code=500, detail=msg)



@register_microservice(name="opea_service@prepare_doc_milvus", endpoint="/v1/knowledge/delete", host="0.0.0.0", port=6010)
async def delete_documents(
        knowledge_name: str = Form(""),
):
    logger.info(f"[ delete ] knowledge_name:{knowledge_name}")

    if not validate_knowledge_name(knowledge_name):
        raise HTTPException(status_code=403, detail="knowledge name format is forbidden")

    if knowledge_name is None or knowledge_name.strip() == "":
        raise HTTPException(status_code=404, detail="knowledge name can't be empty")

    # step 2. check knowledge name is existed or not
    kb = knowledge_service.load_kb_from_db(knowledge_name)
    if kb is None:
        raise HTTPException(status=500, detail="knowledge name has not existed")

    knowledge_name = urllib.parse.unquote(knowledge_name)

    vs = get_vectorstore(knowledge_name=knowledge_name,
                         vs_type=config.vector_store.vector_store_type,
                         embedding_model=embedding_model)

    try:
        # clear vectorstore
        vs.drop_vectorstore()
        # clear files
        delete_kb_folder(knowledge_name)
        # clear db
        status = knowledge_file_service.delete_files_from_db(knowledge_name)
        status2 = knowledge_service.delete_kb_from_db(knowledge_name)
        status3 = url_queue_service.clear_queue(knowledge_name)
        if status and status2 and status3:
            return BaseResponse(status="success", msg=f"delete knowledge success: {knowledge_name}")
    except Exception as e:
        msg = f"delete knowledge error： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)
        return BaseResponse(status="fail", msg=msg)

    return BaseResponse(status="fail", msg=f"delete knowledge fail: {knowledge_name}")


@register_microservice(name="opea_service@prepare_doc_milvus", endpoint="/v1/knowledge/clear", host="0.0.0.0", port=6010)
async def clear_documents(
        knowledge_name: str = Form(""),
):
    logger.info(f"[ clear ] knowledge_name:{knowledge_name}")
    if not validate_knowledge_name(knowledge_name):
        raise HTTPException(status_code=403, detail="knowledge name format is forbidden")

    if knowledge_name is None or knowledge_name.strip() == "":
        raise HTTPException(status_code=404, detail="knowledge name can't be empty")

    # step 2. check knowledge name is existed or not
    kb = knowledge_service.load_kb_from_db(knowledge_name)
    if kb is None:  # 数据库中查不到该数据库
        raise HTTPException(status=500, detail="knowledge name has not existed")

    knowledge_name = urllib.parse.unquote(knowledge_name)
    vs = get_vectorstore(knowledge_name=knowledge_name,
                        vs_type=config.vector_store.vector_store_type,
                        embedding_model=embedding_model)

    try:
        # clear vectorstore
        vs.clear_vectorstore()
        # clear files
        clear_kb_folder(knowledge_name)
        # clear db
        status = knowledge_file_service.delete_files_from_db(knowledge_name)
        status2 = url_queue_service.clear_queue(knowledge_name)
        if status and status2:
            return BaseResponse(status="success", msg=f"clear knowledge success: {knowledge_name}")
    except Exception as e:
        msg = f"clear knowledge error： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)
        return BaseResponse(status="fail", msg=msg)

    return BaseResponse(status="fail", msg=f"clear knowledge fail: {knowledge_name}")




def save_files_in_thread(files: List[UploadFile],
                          knowledge_name: str,
                          override: bool):
    """
    在线程池中保存文件。

    该函数接收文件列表、知识名称和是否覆盖的标志，然后在线程池中异步保存这些文件。

    参数:
    - files: List[UploadFile]，待保存的文件列表。
    - knowledge_name: str，知识名称，用于确定文件保存的目录。
    - override: bool，是否覆盖已存在的文件。

    返回:
    通过生成器逐一返回每个文件保存的结果。
    """

    def save_file(file: UploadFile,
                  knowledge_name: str,
                  override: bool) -> dict:
        """
        保存单个文件。

        该函数尝试保存给定的文件到指定路径，如果文件已存在且不允覆盖，则返回相应的错误信息。

        参数:
        - file: UploadFile，待保存的文件。
        - knowledge_name: str，知识名称，用于确定文件保存的目录。
        - override: bool，是否覆盖已存在的文件。

        返回:
        返回一个字典，包含状态码、消息和文件信息。
        """

        # 获取文件名，用于后续保存和信息返回
        filename = file.filename
        data = {"knowledge_name": knowledge_name, "file_name": filename}
        # 构造文件保存路径
        file_path = get_file_path(knowledge_name=knowledge_name, doc_name=filename)
        try:
            # 读取文件内容
            file_content = file.file.read()
            # 检查文件是否已存在且不允许覆盖，同时文件大小相同，则返回错误信息
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"file {filename} has exsited。"
                logger.warning(file_status)
                return dict(code=404, msg=file_status, data=data)

            # 如果文件目录不存在，则创建目录
            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            # 写入文件内容到指定路径
            with open(file_path, "wb") as f:
                f.write(file_content)
            # 返回成功信息
            return dict(code=200, msg=f"upload file: {filename} success", data=data)
        except Exception as e:
            # 异常处理，返回错误信息
            msg = f"{filename} upload file error: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)
            return dict(code=500, msg=msg, data=data)

    # 构造参数列表，用于线程池处理
    params = [{"file": file, "knowledge_name": knowledge_name, "override": override} for file in files]
    # 在线程池中执行文件保存操作，并逐一返回结果
    for result in run_in_thread_pool(save_file, params=params):
        yield result



def update_docs(
        knowledge_name: str,
        file_names: List[str],
        chunk_size: int,
        chunk_overlap: int,
        smaller_chunk_size: int
) -> BaseResponse:
    """
    更新文档函数，负责将指定的文件更新到指定的知识库中。

    参数:
    - knowledge_name (str): 知识库的名称。
    - file_names (List[str]): 需要更新到知识库中的文件名列表。
    - chunk_size (int): 文档切块的大小。
    - chunk_overlap (int): 相邻文档块之间的重叠大小。
    - smaller_chunk_size (int): 更小的文档切块大小，用于更精细的处理。

    返回:
    - BaseResponse: 包含更新操作结果的响应对象，包括状态码、消息和失败的文件信息。
    """

    # 初始化失败文件字典和知识库文件列表
    failed_files = {}
    kb_files = []

    # 遍历文件名列表，尝试将每个文件加载为KnowledgeFile对象
    for file_name in file_names:
        try:
            # 将文件加载到知识库文件列表中
            kb_files.append(KnowledgeFile(filename=file_name, knowledge_name=knowledge_name))
        except Exception as e:
            # 如果加载过程中出现异常，记录错误信息并将文件名添加到失败文件字典中
            msg = f"加载文档 {file_name} 时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)
            failed_files[file_name] = msg

    # 获取指定知识库的向量存储对象
    vs = get_vectorstore(knowledge_name=knowledge_name,
                        vs_type=config.vector_store.vector_store_type,
                        embedding_model=embedding_model)

    # 创建索引对象，用于处理文档索引
    indexing = Indexing(vectorstore=vs, knowledge_file_service=knowledge_file_service,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        smaller_chunk_size=smaller_chunk_size)

    # 使用索引对象对知识库文件进行索引，并获取失败的文件信息
    failed_files = indexing.index(kb_files)

    # 返回更新文档操作的结果响应
    return BaseResponse(code=200, msg=f"update docs success", data={"failed_files": failed_files})


if __name__ == "__main__":
    # 创建数据库表
    logger.info("创建数据库表...")
    DB.create_tables(DB.get_engine())
    
    # 启动爬虫线程
    logger.info("启动爬虫线程...")
    start_background_tasks()
    # 启动微服务
    logger.info("启动OPEA微服务...")
    opea_microservices["opea_service@prepare_doc_milvus"].start(in_single_process=True)






