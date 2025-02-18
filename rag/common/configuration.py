import os

from .utils import get_env_var

DATA_ROOT_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)))
    ,
    "data"
)


class EmbeddingConfig:
    def __init__(self):
        self.embedding_type = get_env_var("EMBEDDING_TYPE", default="TEI")
        self.mosec_embedding_endpoint = get_env_var("MOSEC_EMBEDDING_ENDPOINT", default="")
        self.mosec_embedding_model = get_env_var("MOSEC_EMBEDDING_MODEL", default="")
        self.tei_embedding_endpoint = get_env_var("TEI_EMBEDDING_ENDPOINT", default="http://127.0.0.1:6006")
        self.local_embedding_model = get_env_var("LOCAL_EMBEDDING_MODEL", default="")


class VectorStoreConfig:
    def __init__(self):
        self.vector_store_type = get_env_var("VECTOR_STORE_TYPE", default="milvus")
        self.milvus = VectorStoreMilvusConfig()


class VectorStoreMilvusConfig:
    def __init__(self):
        self.host = get_env_var("MILVUS_DB_HOST", default="127.0.0.1")
        self.port = get_env_var("MILVUS_DB_PORT", default=19530, cast=int)
        self.user = get_env_var("MILVUS_DB_USER", default="")
        self.password = get_env_var("MILVUS_DB_PASSWORD", default="")
        self.db_name = get_env_var("MILVUS_DB_NAME", default="default")
        self.kwargs = {
            "search_params": {"metric_type": "IP"},
            "index_params": {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}
        }


class LLMConfig:
    def __init__(self):
        self.host = get_env_var("LLM_HOST", default="localhost")
        self.port = get_env_var("LLM_PORT", default=8000, cast=int)
        self.user = get_env_var("LLM_USER", default="root")
        self.password = get_env_var("LLM_PASSWORD", default="password")


class ServerConfig:
    def __init__(self):
        self.host = get_env_var("SERVER_HOST", default="localhost")
        self.port = get_env_var("SERVER_PORT", default=8000, cast=int)


class DatabaseConfig:
    def __init__(self):
        self.sqlalchemy_database_uri = get_env_var("SQLALCHEMY_DATABASE_URI", default="")


class SplitterConfig:
    def __init__(self):
        self.chunk_size = get_env_var("CHUNK_SIZE", default=512, cast=int)
        self.chunk_overlap = get_env_var("CHUNK_OVERLAP", default=100, cast=int)
        self.smaller_chunk_size = get_env_var("SMALLER_CHUNK_SIZE", default=0, cast=int)
        self.splitter_name = get_env_var("SPLITTER_NAME", default="ChineseRecursiveTextSplitter")


class Configuration:
    def __init__(self):
        self.embedding = EmbeddingConfig()
        self.vector_store = VectorStoreConfig()
        self.llm = LLMConfig()
        self.server = ServerConfig()
        self.database = DatabaseConfig()
        self.data_root_path = get_env_var("DATA_ROOT_PATH", default=DATA_ROOT_PATH)
        self.splitter = SplitterConfig()


config = Configuration()
