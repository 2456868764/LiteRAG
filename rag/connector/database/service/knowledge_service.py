from comps.cores.mega.logger import CustomLogger
from rag.connector.database.models.knowledge_base_model import KnowledgeBaseModel
from rag.connector.database.session import with_session
from typing import Optional, List, Tuple
from rag.connector.database.service.url_queue_service import URLQueueService


logger = CustomLogger("knowledge service")

class KnowledgeService:
    def __init__(self):
        self.url_queue_service = URLQueueService()

    @with_session
    def list_kbs_from_db(self, session, min_file_count: int = -1) -> List[str]:
        """获取知识库列表"""
        kbs = session.query(KnowledgeBaseModel.kb_name).filter(
            KnowledgeBaseModel.file_count > min_file_count
        ).all()
        return [kb[0] for kb in kbs]

    @with_session
    def add_kb_to_db(
        self, 
        session, 
        kb_name: str,
        kb_info: str,
        vs_type: str,
        embed_model: str,
        weburl: Optional[str] = "",
        scraping_level: int = 1,
        link_tags: Optional[str] = ""
    ) -> bool:
        """添加或更新知识库"""
        kb = session.query(KnowledgeBaseModel).filter(
            KnowledgeBaseModel.kb_name.ilike(kb_name)
        ).first()

        if not kb:
            kb = KnowledgeBaseModel(
                kb_name=kb_name,
                kb_info=kb_info,
                vs_type=vs_type,
                embed_model=embed_model,
                weburl=weburl,
                scraping_level=scraping_level,
                link_tags=link_tags
            )
            session.add(kb)
        else:
            kb.kb_info = kb_info
            kb.vs_type = vs_type
            kb.embed_model = embed_model
            kb.weburl = weburl
            kb.scraping_level = scraping_level
            kb.link_tags = link_tags

        # 如果提供了weburl，添加到爬取队列
        if weburl and scraping_level > 0:
            logger.info(f"add url to queue: {weburl}, scraping_level: {scraping_level}, link_tags: {link_tags}")
            self.url_queue_service.add_url_to_queue(kb_name, weburl, scraping_level, link_tags)

        return True

    @with_session
    def load_kb_from_db(self, session, kb_name: str) -> Tuple[Optional[str], ...]:
        """加载知识库信息"""
        kb = session.query(KnowledgeBaseModel).filter(
            KnowledgeBaseModel.kb_name.ilike(kb_name)
        ).first()
        
        if kb:
            return {
                'kb_name': kb.kb_name,
                'kb_info': kb.kb_info,
                'vs_type': kb.vs_type,
                'embed_model': kb.embed_model,
                'weburl': kb.weburl,
                'scraping_level': kb.scraping_level,
                'link_tags': kb.link_tags,
                'file_count': kb.file_count
            }
        return None

    @with_session
    def delete_kb_from_db(self, session, kb_name: str) -> bool:
        """删除知识库"""
        kb = session.query(KnowledgeBaseModel).filter(
            KnowledgeBaseModel.kb_name.ilike(kb_name)
        ).first()
        if kb:
            session.delete(kb)
        return True

    @with_session
    def update_kb_file_count(self, session, kb_name: str, count: int) -> bool:
        """更新知识库文件数量"""
        kb = session.query(KnowledgeBaseModel).filter(
            KnowledgeBaseModel.kb_name.ilike(kb_name)
        ).first()
        if kb:
            kb.file_count = count
            return True
        return False