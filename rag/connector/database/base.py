import os
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from rag.common.configuration import config

Base = declarative_base()

# 数据库默认存储路径
KNOWLEDGE_ROOT_PATH = os.path.join(
    config.data_root_path,
    "database"
)

# 确保知识库目录存在
if not os.path.exists(KNOWLEDGE_ROOT_PATH):
    os.makedirs(KNOWLEDGE_ROOT_PATH)

if config.database.sqlalchemy_database_uri == "":
    DB_ROOT_PATH = os.path.join(KNOWLEDGE_ROOT_PATH, "kb.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"
else:
    SQLALCHEMY_DATABASE_URI = config.database.sqlalchemy_database_uri

print(f"sql uri:{SQLALCHEMY_DATABASE_URI}")
class DB:
    @staticmethod
    def get_engine():
        """获取数据库引擎"""
        return create_engine(
            SQLALCHEMY_DATABASE_URI,
            json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
        )

    @staticmethod
    def create_tables(engine):
        """创建数据库表格"""
        Base.metadata.create_all(bind=engine)

    @staticmethod
    def get_session():
        """返回一个数据库会话实例"""
        return SessionLocal()


# 获取数据库引擎和会话类
engine = DB.get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
