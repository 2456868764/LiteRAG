from sqlalchemy import Column, Integer, String, DateTime, func
from rag.connector.database.base import Base
from datetime import datetime

class KnowledgeBaseModel(Base):
    """
    知识库模型
    """
    __tablename__ = 'knowledge_base'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库ID')
    kb_name = Column(String(50), comment='知识库名称')
    kb_info = Column(String(200), comment='知识库简介')
    vs_type = Column(String(50), comment='向量库类型')
    embed_model = Column(String(50), comment='嵌入模型类型')
    file_count = Column(Integer, default=0, comment='文件数量')
    weburl = Column(String(500), comment='网页URL')
    scraping_level = Column(Integer, default=1, comment='URL爬取深度')
    link_tags = Column(String(500), comment='链接标签，用于过滤爬取的URL，多个标签用逗号分隔')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')

    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info} vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', weburl='{self.weburl}', scraping_level='{self.scraping_level}', create_time='{self.create_time}')>"
