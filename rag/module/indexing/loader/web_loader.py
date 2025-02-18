from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import trafilatura
from urllib.parse import urlparse
import chardet
from comps.cores.mega.logger import CustomLogger

logger = CustomLogger("webbase_loader")

class CustomizedWebBaseLoader(BaseLoader):
    """
    自定义网页加载器，支持更好的内容提取和清理，特别优化中文网页支持
    """
    
    def __init__(
        self,
        web_path: str,
        header_template: Optional[dict] = None,
        verify_ssl: bool = True,
        custom_parser: bool = True,
        remove_selectors: List[str] = None,
        timeout: int = 20,
        max_retries: int = 3
    ):
        """
        初始化加载器
        Args:
            web_path: 网页URL
            header_template: 请求头模板
            verify_ssl: 是否验证SSL证书
            custom_parser: 是否使用自定义解析（使用trafilatura）
            remove_selectors: 要移除的CSS选择器列表
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
        """
        self.web_path = web_path
        self.verify = verify_ssl
        self.custom_parser = custom_parser
        self.timeout = timeout
        self.max_retries = max_retries
        self.remove_selectors = remove_selectors or [
            '.advertisement', '#ads', '.ads',  # 广告
            '.social-share', '.social-media',  # 社交分享
            '.comment-section', '#comments',  # 评论区
            'nav'
        ]
        
        # 设置默认请求头，添加中文语言支持
        self.header_template = header_template or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        # 创建session
        self.session = requests.Session()
        self.session.headers.update(self.header_template)

    def _detect_encoding(self, content: bytes) -> str:
        """
        检测内容编码
        """
        result = chardet.detect(content)
        return result['encoding'] or 'utf-8'

    def _fetch_with_retry(self, url: str) -> requests.Response:
        """
        带重试机制的URL获取
        Args:
            url: 要获取的URL
        Returns:
            Response对象
        Raises:
            requests.RequestException: 当所有重试都失败时抛出
        """
        retry_count = 0
        last_exception = None

        while retry_count < self.max_retries:
            try:
                logger.info(f"尝试获取URL (第 {retry_count + 1} 次): {url}")
                response = self.session.get(
                    url,
                    verify=self.verify,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"成功获取URL: {url}")
                    # 检测并设置正确的编码
                    if not response.encoding or response.encoding.lower() == 'iso-8859-1':
                        response.encoding = self._detect_encoding(response.content)
                    return response
                    
                response.raise_for_status()
                
            except requests.RequestException as e:
                last_exception = e
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"获取失败，准备第 {retry_count + 1} 次重试: {str(e)}")
                    continue
                
        error_msg = f"重试 {self.max_retries} 次后失败"
        logger.error(error_msg)
        raise last_exception or requests.RequestException(error_msg)

    def _extract_code(self, tag) -> str:
        """
        提取代码块内容，保持格式
        """
        # 获取代码语言（如果有）
        language = tag.get('class', [''])[0] if tag.get('class') else ''
        if language.startswith('language-'):
            language = language[9:]
        elif language.startswith('lang-'):
            language = language[5:]
        
        code_content = tag.get_text(strip=False)  # 保持原始格式
        
        # 如果有语言标识，添加到代码块
        if language:
            return f"\n```{language}\n{code_content}\n```\n"
        return f"\n```\n{code_content}\n```\n"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        提取文本内容，保持结构，特别处理代码块
        """
        logger.info("开始提取文本内容...")
        text_parts = []
        
        # 提取标题
        if soup.title:
            title_text = soup.title.get_text(strip=True)
            logger.info(f"提取标题: {title_text}")
            text_parts.append(f"标题: {title_text}\n")
        
        # 遍历所有元素
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'code']):
            # 如果是代码块
            if tag.name == 'pre' or (tag.name == 'code' and not tag.parent.name == 'pre'):
                logger.debug("处理代码块...")
                text = self._extract_code(tag)
            else:
                text = tag.get_text(strip=True)
                if text and tag.name.startswith('h'):
                    logger.debug(f"处理标题: {text}")
                    text = f"\n{text}\n"
            
            if text:
                text_parts.append(text)
        
        result = '\n'.join(filter(None, text_parts))
        logger.info(f"文本提取完成，总长度: {len(result)}")
        return result

    def _clean_html(self, html_content: str) -> str:
        """
        清理HTML内容，保留代码块
        """
        soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        
        # 移除不需要的元素，但保留代码块
        for selector in self.remove_selectors:
            for element in soup.select(selector):
                # 检查是否包含代码块
                if not element.find(['pre', 'code']):
                    element.decompose()
        
        # 只移除不在代码块内的script和style标签
        for tag in soup.find_all(['script', 'style']):
            if not tag.find_parent(['pre', 'code']):
                tag.decompose()
        
        return str(soup)

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """
        提取页面元数据
        """
        metadata = {
            'url': url,
            'title': self._extract_title(soup),
            'domain': urlparse(url).netloc
        }
        
        # 提取meta标签信息，支持中文编码
        meta_tags = {
            'description': ['description', 'og:description'],
            'keywords': ['keywords'],
            'author': ['author', 'og:author'],
            'published_time': ['article:published_time', 'publishedTime']
        }
        
        for key, meta_names in meta_tags.items():
            for name in meta_names:
                meta = soup.find('meta', attrs={'name': name}) or soup.find('meta', attrs={'property': name})
                if meta and meta.get('content'):
                    metadata[key] = meta['content']
                    break
        
        return metadata

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        提取页面标题
        """
        title = soup.find('title')
        if title:
            return title.get_text(strip=True)
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def _detect_language(self, soup: BeautifulSoup) -> str:
        """
        从HTML中检测页面语言
        Returns:
            str: 语言代码 (如 'zh', 'en' 等)
        """
        # 从 html 标签的 lang 属性检测
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            lang = html_tag.get('lang').lower()
            return lang.split('-')[0]  # 'zh-CN' -> 'zh'
        
        # 从 meta 标签检测
        meta_lang = soup.find('meta', attrs={'http-equiv': 'content-language'})
        if meta_lang and meta_lang.get('content'):
            return meta_lang['content'].lower().split('-')[0]
        
        # 从 og:locale 检测
        og_locale = soup.find('meta', property='og:locale')
        if og_locale and og_locale.get('content'):
            return og_locale['content'].lower().split('_')[0]
        
        # 默认返回英语
        return 'en'

    def load(self) -> List[Document]:
        """
        加载网页内容
        Returns:
            List[Document]: 文档列表
        """
        docs = []
        for url in self.web_path if isinstance(self.web_path, list) else [self.web_path]:
            try:
                logger.info(f"开始加载URL: {url}")
                
                # 获取页面内容
                logger.info("发送HTTP请求...")
                response = self._fetch_with_retry(url)
                html_content = response.text
                logger.info(f"页面编码: {response.encoding}")
                
                # 检测语言
                logger.info("检测页面语言...")
                soup = BeautifulSoup(html_content, 'html.parser', from_encoding=response.encoding)
                target_lang = self._detect_language(soup)
                logger.info(f"检测到语言: {target_lang}")
                
                # 提取内容
                if self.custom_parser:
                    logger.info("使用trafilatura提取主要内容...")
                    text = trafilatura.extract(
                        html_content,
                        include_links=True,
                        include_images=True,
                        include_formatting=True,
                        target_language=target_lang
                    )
                    if not text:
                        logger.warning("trafilatura提取失败，回退到BeautifulSoup")
                        text = self._extract_text(soup)
                else:
                    logger.info("使用BeautifulSoup提取内容...")
                    cleaned_html = self._clean_html(html_content)
                    soup = BeautifulSoup(cleaned_html, 'html.parser', from_encoding=response.encoding)
                    text = self._extract_text(soup)
                
                # 提取元数据
                logger.info("提取页面元数据...")
                metadata = self._extract_metadata(soup, url)
                metadata['language'] = target_lang
                
                # 创建文档
                logger.info("创建文档对象...")
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                docs.append(doc)
                
                logger.info(f"成功加载文档，内容长度: {len(text)}")
                logger.debug(f"元数据: {metadata}")
                
            except requests.RequestException as e:
                logger.error(f"获取 {url} 失败 (重试 {self.max_retries} 次): {str(e)}")
                continue
            except Exception as e:
                logger.error(f"处理 {url} 时出错: {str(e)}", exc_info=True)
                continue
                
        return docs


def test_loader():
    """
    测试网页加载器
    """
    # 测试URL
    test_url = "https://istio.io/latest/docs/overview/what-is-istio/"
    test_url = "https://istio.io/latest/docs/concepts/traffic-management/"
    # test_url = "https://higress.cn/docs/latest/overview/what-is-higress/"
    # test_url = "https://higress.cn/docs/latest/ops/how-tos/gateway-ports"
    
    try:
        # 初始化加载器
        loader = CustomizedWebBaseLoader(
            web_path=test_url,
            custom_parser=True
        )
        
        # 加载文档
        docs = loader.load()
        
        # 打印结果
        print(f"\n成功加载文档数量: {len(docs)}")
        if docs:
            doc = docs[0]
            print("\n文档元数据:")
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")
            
            print("\n文档内容片段:")
            content_preview = doc.page_content
            print(content_preview)
            
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")


if __name__ == "__main__":
    test_loader() 