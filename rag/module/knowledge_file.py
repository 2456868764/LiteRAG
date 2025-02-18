import os
import shutil
from pathlib import Path
from typing import Literal

from rag.common.configuration import config
from rag.module.indexing.loader import DOCUMENTS_LOADER_MAPPING
from rag.module.indexing.splitter import ChineseTextSplitter, DOCUMENTS_SPLITER_MAPPING
from rag.module.utils import get_loader


DOC_ROOT_PATH = os.path.join(
    config.data_root_path,
    "uploaded_files"
)

if not os.path.exists(DOC_ROOT_PATH):
    os.makedirs(DOC_ROOT_PATH)


def get_kb_path(knowledge_name: str):
    kb_path = os.path.join(DOC_ROOT_PATH, knowledge_name)
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
    return kb_path


def get_file_path(knowledge_name: str, doc_name: str):
    return os.path.join(get_kb_path(knowledge_name), doc_name)


def clear_kb_folder(knowledge_name: str):
    kb_path = os.path.join(DOC_ROOT_PATH, knowledge_name)
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)  # 删除文件夹及其内容
        os.makedirs(kb_path)  # 重新创建空文件夹


def delete_kb_folder(knowledge_name: str):
    kb_path = os.path.join(DOC_ROOT_PATH, knowledge_name)
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)  # 删除文件夹及其内容


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_name: str,
            file_type: Literal['file', 'url'] = 'file'
    ):
        self.knowledge_name = knowledge_name
        self.type = file_type
        if file_type == 'file':
            self.filename = str(Path(filename).as_posix())
            self.ext = os.path.splitext(filename)[-1].lower()
            self.filepath = filename if os.path.exists(filename) else get_file_path(knowledge_name, filename)
        else:
            self.filename = filename
            self.ext = ''
            self.filepath = filename
        
        self.document_loader = self.get_document_loader()
        self.text_splitter = self.get_text_splitter()

    def get_type(self) -> str:
        """
        获取知识文件类型
        Returns:
            str: 'file' 或 'url'
        """
        return self.type

    def get_document_loader(self):
        if self.type == 'url':
            return get_loader('CustomizedWebBaseLoader')
            
        loader_name = ""
        for loader_cls, extensions in DOCUMENTS_LOADER_MAPPING.items():
            if self.ext in extensions:
                loader_name = loader_cls
                break
        return get_loader(loader_name)

    def get_text_splitter(self):
        if config.splitter.splitter_name not in DOCUMENTS_SPLITER_MAPPING:
            return ChineseTextSplitter
        return DOCUMENTS_SPLITER_MAPPING[config.splitter.splitter_name]

    def file_exist(self):
        if self.type == 'url':
            return True
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        if self.type == 'url':
            return 0.0
        return os.path.getmtime(self.filepath)

    def get_size(self):
        if self.type == 'url':
            return 0
        return os.path.getsize(self.filepath)