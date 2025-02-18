import time
from typing import List, Set
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from comps.cores.mega.logger import CustomLogger
from rag.connector.database.base import DB
from rag.connector.database.service.url_queue_service import URLQueueService
from rag.connector.database.models.url_queue_model import URLStatus
from rag.connector.utils import get_embedding_model, get_vectorstore
from rag.module.indexing.indexing import Indexing
from rag.module.knowledge_file import KnowledgeFile
from rag.connector.database.service.knowledge_file_service import KnowledgeFileService
from rag.common.configuration import config

logger = CustomLogger("url_crawler")


# get embedding model
embedding_model = get_embedding_model(embedding_type=config.embedding.embedding_type,
                                        mosec_embedding_model=config.embedding.mosec_embedding_model,
                                        mosec_embedding_endpoint=config.embedding.mosec_embedding_endpoint,
                                        tei_embedding_endpoint=config.embedding.tei_embedding_endpoint,
                                        local_embedding_model=config.embedding.local_embedding_model)



class URLCrawler:
    def __init__(self):
        self.url_queue_service = URLQueueService()
        self.file_service = KnowledgeFileService()

    def get_internal_links(self, base_url: str, link_tags: str, html_content: str) -> List[str]:
        """
        获取页面中的站内链接
        Args:
            base_url: 基础URL
            link_tags: 选择器字符串，支持以下格式：
                      - HTML标签: 'nav,main,article'
                      - CSS类: '.menu,.content,.sidebar'
                      - ID: '#header,#footer,#main-content'
                      多个选择器用逗号分隔
            html_content: 页面HTML内容
        Returns:
            List[str]: 站内链接列表
        """
        logger.info(f"开始从 {base_url} 提取链接 (选择器: {link_tags or '所有'})")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = urlparse(base_url).netloc
        internal_links = set()
        
        if link_tags:
            # 处理多个选择器，以逗号分隔
            selectors = [selector.strip() for selector in link_tags.split(',')]
            logger.debug(f"处理选择器: {selectors}")
            
            for selector in selectors:
                try:
                    if selector.startswith('.'):  # CSS类
                        logger.debug(f"查找CSS类: {selector[1:]}")
                        elements = soup.find_all(class_=selector[1:])
                    elif selector.startswith('#'):  # ID
                        logger.debug(f"查找ID: {selector[1:]}")
                        element = soup.find(id=selector[1:])
                        elements = [element] if element else []
                    else:  # HTML标签
                        logger.debug(f"查找标签: {selector}")
                        elements = soup.find_all(selector)
                    
                    logger.debug(f"使用选择器 '{selector}' 找到 {len(elements)} 个元素")
                    
                    # 从找到的元素中提取链接
                    for element in elements:
                        if element:  # 确保元素存在
                            for link in element.find_all('a', href=True):
                                url = urljoin(base_url, link['href'])
                                if urlparse(url).netloc == base_domain:
                                    internal_links.add(url)
                                    logger.debug(f"在 {selector} 中找到链接: {url}")
                                    
                except Exception as e:
                    logger.error(f"处理选择器 {selector} 时出错: {str(e)}")
                    continue
        else:
            # 如果没有指定选择器，获取所有链接
            logger.debug("获取所有链接")
            for link in soup.find_all('a', href=True):
                url = urljoin(base_url, link['href'])
                if urlparse(url).netloc == base_domain:
                    internal_links.add(url)
        
        logger.info(f"找到 {len(internal_links)} 个站内链接")
        for url in internal_links:
            logger.debug(f"发现链接: {url}")
                
        return list(internal_links)

    def process_url(self, url_info: dict) -> None:
        """处理单个URL"""
        logger.info(f"开始处理 URL: {url_info['url']}")
        logger.info(f"知识库: {url_info['kb_name']}, 剩余深度: {url_info['scraping_level']}")
        
        try:
            # 更新状态为处理中
            logger.debug("更新状态为处理中...")
            self.url_queue_service.update_url_status(url_info['id'], URLStatus.RUNNING)
            
            # 创建KnowledgeFile实例
            logger.debug("创建知识库文件...")
            self.process_vector_store(url_info)
            # 如果还有爬取深度，获取并添加内部链接
            if url_info['scraping_level'] > 0:
                logger.info(f"获取页面内容: {url_info['url']}")
                response = requests.get(url_info['url'])
                response.encoding = response.apparent_encoding
                
                logger.info("提取内部链接...")
                internal_links = self.get_internal_links(
                    url_info['url'], 
                    url_info['link_tags'],
                    response.text
                )
                
                # 添加新的URL到队列
                logger.info(f"添加 {len(internal_links)} 个新URL到队列")
                for link in internal_links:
                    self.url_queue_service.add_url_to_queue(
                        url_info['kb_name'],
                        link,
                        url_info['scraping_level'] - 1,
                        url_info['link_tags']
                    )
            
            # 更新状态为完成
            logger.info("处理完成")
            self.url_queue_service.update_url_status(url_info['id'], URLStatus.COMPLETED)
            
        except Exception as e:
            # 更新状态为失败
            logger.error(f"处理失败: {str(e)}", exc_info=True)
            self.url_queue_service.update_url_status(
                url_info['id'],
                URLStatus.FAILED,
                str(e)
            )
            
    def process_vector_store(self, url_info: dict):
        """处理单个URL"""
        logger.info(f"开始索引 URL vector store: {url_info['url']} in kb_name: {url_info['kb_name']}")
        try:
            # 更新状态为处理中
            logger.info("索引处理中...")
            # 初始化失败文件字典和知识库文件列表
            failed_files = {}
            kb_files = []
            knowledge_name = url_info['kb_name']
            # 遍历文件名列表，尝试将每个文件加载为KnowledgeFile对象
            # 将文件加载到知识库文件列表中
            kb_files.append(KnowledgeFile(filename=url_info['url'], knowledge_name=url_info['kb_name'], file_type='url'))
            # 如果加载过程中出现异常，记录错误信息并将文件名添加到失败文件字典中
            # 获取指定知识库的向量存储对象
            vs = get_vectorstore(knowledge_name=knowledge_name,
                                vs_type=config.vector_store.vector_store_type,
                                embedding_model=embedding_model)
            # 创建索引对象，用于处理文档索引
            indexing = Indexing(vectorstore=vs, knowledge_file_service=self.file_service,
                                chunk_size=config.splitter.chunk_size,
                                chunk_overlap=config.splitter.chunk_overlap,
                                smaller_chunk_size=config.splitter.smaller_chunk_size)

            # 使用索引对象对知识库文件进行索引，并获取失败的文件信息
            failed_files = indexing.index(kb_files)
            logger.info(f"索引完成，失败文件: {failed_files}")
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}", exc_info=True)


    def run(self, batch_size: int = 5):
        """运行爬虫任务"""
        start_time = time.time()
        logger.info(f"启动爬虫 (批次大小: {batch_size})")
        
        logger.debug("获取待处理URL...")
        pending_urls = self.url_queue_service.get_pending_urls(limit=batch_size)
        logger.info(f"获取到 {len(pending_urls)} 个待处理URL")
        
        if len(pending_urls) == 0:
            logger.info("没有待处理的URL，等待5秒...")
            return False
        
        # 处理每个URL
        for url_info in pending_urls:
            url_start_time = time.time()
            self.process_url(url_info)
            url_end_time = time.time()
            logger.info(f"处理URL耗时: {url_end_time - url_start_time:.2f}秒")

        end_time = time.time()
        logger.info(f"本批次处理完成，总耗时: {end_time - start_time:.2f}秒")
        return True
            

def test_crawler():
    """
    测试URL爬虫功能
    """
    crawler = URLCrawler()
    
    test_cases = [
        {
            "name": "Istio文档爬取测试",
            "kb_name": "test_istio",
            "url": "https://istio.io/latest/docs/",
            "scraping_level": 1,
            "link_tags": "nav",
            "description": "测试Istio文档页面爬取，只获取文章内容区域的链接"
        },
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试用例: {case['name']}")
        print(f"URL: {case['url']}")
        print(f"描述: {case['description']}")
        print(f"爬取深度: {case['scraping_level']}")
        print(f"链接标签: {case['link_tags'] or '所有'}")
        print('='*50)

        try:
            # 清空之前的队列
            crawler.url_queue_service.clear_queue(case['kb_name'])
            
            # 添加初始URL到队列
            crawler.url_queue_service.add_url_to_queue(
                kb_name=case['kb_name'],
                url=case['url'],
                scraping_level=case['scraping_level'],
                link_tags=case['link_tags']
            )
            
            # 运行爬虫（只处理一批）
            print("\n开始爬取...")
            crawler.run(batch_size=1)
            
            # 获取统计信息
            stats = crawler.url_queue_service.get_queue_stats(case['kb_name'])
            print("\n爬取结果:")
            print(f"总URL数: {stats['total']}")
            print(f"完成数: {stats['completed']}")
            print(f"待处理数: {stats['pending']}")
            print(f"失败数: {stats['failed']}")
            
            # 获取已完成的URL列表
            completed_urls = crawler.url_queue_service.get_completed_urls(case['kb_name'])
            if completed_urls:
                print("\n成功爬取的URL:")
                for url_info in completed_urls:
                    print(f"- {url_info['url']}")
            
            # 获取失败的URL列表
            failed_urls = crawler.url_queue_service.get_failed_urls(case['kb_name'])
            if failed_urls:
                print("\n失败的URL:")
                for url_info in failed_urls:
                    print(f"- {url_info['url']}: {url_info['error']}")
            
            # 检查知识库文件
            kb_files = crawler.file_service.get_files_by_kb_name(case['kb_name'])
            if kb_files:
                print("\n创建的知识库文件:")
                for file in kb_files:
                    print(f"- {file.filename}")
            
            # 特定场景的断言
            if case['name'] == "错误处理测试":
                assert stats['failed'] > 0, "错误处理测试应该有失败记录"
            else:
                assert stats['completed'] > 0, "没有成功爬取的URL"
                assert stats['failed'] == 0, "存在爬取失败的URL"
            
            print("\n✓ 测试通过")
            
        except Exception as e:
            print(f"\n✗ 测试失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            # 清理测试数据
            crawler.url_queue_service.clear_queue(case['kb_name'])
            crawler.file_service.delete_files_from_db(case['kb_name'])
            crawler.file_service.delete_docs_from_db(case['kb_name'])


if __name__ == "__main__":
    print("运行URL爬虫测试...")
    logger.info("create tables")
    DB.create_tables(DB.get_engine())
    test_crawler()

