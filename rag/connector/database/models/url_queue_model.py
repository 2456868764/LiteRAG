from sqlalchemy import Column, Integer, String, DateTime, Enum, func
from rag.connector.database.base import Base
import enum


class URLStatus(enum.Enum):
    PENDING = "pending"    # 等待爬取
    RUNNING = "running"    # 正在爬取
    COMPLETED = "completed"  # 爬取完成
    FAILED = "failed"      # 爬取失败


class URLQueueModel(Base):
    """
    URL爬取队列模型
    """
    __tablename__ = 'url_queue'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='队列ID')
    kb_name = Column(String(50), comment='所属知识库名称')
    url = Column(String(500), comment='待爬取URL')
    scraping_level = Column(Integer, default=0, comment='剩余爬取深度')
    link_tags = Column(String(500), comment='链接标签，用于过滤子URL，多个标签用逗号分隔')
    status = Column(Enum(URLStatus), default=URLStatus.PENDING, comment='爬取状态')
    error_msg = Column(String(500), nullable=True, comment='错误信息')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')

    def __repr__(self):
        return f"<URLQueue(id='{self.id}', kb_name='{self.kb_name}', url='{self.url}', scraping_level='{self.scraping_level}', link_tags='{self.link_tags}', status='{self.status}')>" 