from typing import List, Optional
from sqlalchemy import and_
from rag.connector.database.models.url_queue_model import URLQueueModel, URLStatus
from rag.connector.database.session import with_session
import re


class URLQueueService:
    """
    URL爬取队列服务
    """
    
    @with_session
    def add_url_to_queue(
        self, 
        session,
        kb_name: str,
        url: str,
        scraping_level: int,
        link_tags: Optional[str] = None
    ) -> bool:
        """
        添加URL到爬取队列
        Args:
            kb_name: 知识库名称
            url: 待爬取URL
            scraping_level: 剩余爬取深度
            link_tags: URL过滤标签(可选)
        """
        # 检查URL是否已在队列中
        existing = session.query(URLQueueModel).filter(
            and_(
                URLQueueModel.kb_name == kb_name,
                URLQueueModel.url == url
            )
        ).first()
        
        if not existing:
            # 添加新URL到队列
            queue_item = URLQueueModel(
                kb_name=kb_name,
                url=url,
                scraping_level=scraping_level,
                link_tags=link_tags,  # 保存link_tags到数据库
                status=URLStatus.PENDING
            )
            session.add(queue_item)
            return True
            
        return False

    @with_session
    def get_pending_urls(
        self,
        session,
        kb_name: Optional[str] = None,
        limit: int = 10
    ) -> List[dict]:
        """
        获取待处理的URL列表
        Args:
            kb_name: 知识库名称（可选，用于过滤特定知识库的URL）
            limit: 返回的最大URL数量
        Returns:
            List[dict]: 包含待处理URL信息的字典列表
        """
        query = session.query(URLQueueModel).filter(
            URLQueueModel.status == URLStatus.PENDING
        )
        
        if kb_name:
            query = query.filter(URLQueueModel.kb_name == kb_name)
            
        pending = query.limit(limit).all()
        # 在session活跃时获取所有需要的数据
        return [{
            'id': item.id,
            'url': item.url,
            'kb_name': item.kb_name,
            'scraping_level': item.scraping_level,
            'link_tags': item.link_tags,
            'status': item.status
        } for item in pending]

    @with_session
    def update_url_status(
        self,
        session,
        url_id: int,
        status: URLStatus,
        error_msg: Optional[str] = None
    ) -> bool:
        """
        更新URL状态
        Args:
            url_id: URL队列项ID
            status: 新状态
            error_msg: 错误信息（如果有）
        """
        queue_item = session.query(URLQueueModel).get(url_id)
        if queue_item:
            queue_item.status = status
            if error_msg:
                queue_item.error_msg = error_msg
            return True
        return False

    @with_session
    def clear_queue(self, session, kb_name: Optional[str] = None) -> bool:
        """
        清空爬取队列
        Args:
            kb_name: 知识库名称（可选，仅清空特定知识库的队列）
        """
        query = session.query(URLQueueModel)
        if kb_name:
            query = query.filter(URLQueueModel.kb_name == kb_name)
        query.delete()
        return True

    @with_session
    def get_completed_urls(self, session, kb_name: str) -> List[dict]:
        """
        获取已完成的URL列表
        Args:
            kb_name: 知识库名称
        Returns:
            List[dict]: 包含URL信息的字典列表
        """
        completed = session.query(URLQueueModel).filter(
            and_(
                URLQueueModel.kb_name == kb_name,
                URLQueueModel.status == URLStatus.COMPLETED
            )
        ).all()
        # 在session活跃时获取所有需要的数据
        return [{'url': item.url, 'id': item.id} for item in completed]

    @with_session
    def get_failed_urls(self, session, kb_name: str) -> List[dict]:
        """
        获取失败的URL列表及错误信息
        Args:
            kb_name: 知识库名称
        Returns:
            List[dict]: 包含URL和错误信息的字典列表
        """
        failed = session.query(URLQueueModel).filter(
            and_(
                URLQueueModel.kb_name == kb_name,
                URLQueueModel.status == URLStatus.FAILED
            )
        ).all()
        # 在session活跃时获取所有需要的数据
        return [{'url': item.url, 'error': item.error_msg, 'id': item.id} for item in failed]

    @with_session
    def get_queue_stats(self, session, kb_name: Optional[str] = None) -> dict:
        """
        获取队列统计信息
        Args:
            kb_name: 知识库名称（可选）
        Returns:
            dict: 包含各状态URL数量的统计信息
        """
        query = session.query(URLQueueModel)
        if kb_name:
            query = query.filter(URLQueueModel.kb_name == kb_name)
            
        total = query.count()
        stats = {
            'total': total,
            'pending': query.filter(URLQueueModel.status == URLStatus.PENDING).count(),
            'running': query.filter(URLQueueModel.status == URLStatus.RUNNING).count(),
            'completed': query.filter(URLQueueModel.status == URLStatus.COMPLETED).count(),
            'failed': query.filter(URLQueueModel.status == URLStatus.FAILED).count()
        }
        return stats
