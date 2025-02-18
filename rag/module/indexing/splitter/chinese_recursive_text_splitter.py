import re
from typing import List, Optional, Any, Iterable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    if separator:
        if keep_separator:
            # 根据分隔符对文本进行拆分, 并且保留分隔符
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # 取第一个为默认分隔符，剩余为候选分隔符
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents[:1]:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas) + documents[1:]

def test_splitter():
    """
    测试中文分词器的各种场景
    """
    # 创建分词器实例
    splitter = ChineseRecursiveTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        keep_separator=True,
    )

    test_cases = [
        {
            "name": "中文段落测试",
            "text": """
            自然语言处理是人工智能的一个重要分支。它研究人与计算机之间用自然语言进行有效通信的各种理论和方法。
            自然语言处理的应用非常广泛，包括：机器翻译、信息检索、问答系统、对话系统等。
            近年来，深度学习技术的发展使自然语言处理取得了重大进展。
            """,
            "expected_chunks": 3
        },
        {
            "name": "中英混合测试",
            "text": """
            RAG (Retrieval Augmented Generation) 是一种新型的生成式AI架构。
            It combines the power of large language models with external knowledge retrieval.
            这种方法可以显著提升模型的准确性和可靠性。
            The key advantage is its ability to access up-to-date information.
            """,
            "expected_chunks": 4
        },
        {
            "name": "代码块测试",
            "text": """
            以下是一个Python代码示例：
            ```python
            def hello_world():
                print("Hello, World!")
                return True
            ```
            这个函数会打印Hello, World!并返回True。
            """,
            "expected_chunks": 2
        },
        {
            "name": "特殊分隔符测试",
            "text": """
            第一部分！这是一个测试句子。
            第二部分？这也是测试句子。
            第三部分；还是测试句子。
            第四部分，继续测试。
            """,
            "expected_chunks": 4
        },

        {
            "name": "istio",
            "text": """
            Istio’s traffic management model relies on the Envoy proxies that are deployed along with your services. All traffic that your mesh services send and receive (data plane traffic) is proxied through Envoy, making it easy to direct and control traffic around your mesh without making any changes to your services.

If you’re interested in the details of how the features described in this guide
work, you can find out more about Istio’s traffic management implementation in the
[architecture overview](/latest/docs/ops/deployment/architecture/). The rest of
this guide introduces Istio’s traffic management features.

## Introducing Istio traffic management

In order to direct traffic within your mesh, Istio needs to know where all your endpoints are, and which services they belong to. To populate its own service registry, Istio connects to a service discovery system. For example, if you’ve installed Istio on a Kubernetes cluster, then Istio automatically detects the services and endpoints in that cluster.

Using this service registry, the Envoy proxies can then direct traffic to the relevant services. Most microservice-based applications have multiple instances of each service workload to handle service traffic, sometimes referred to as a load balancing pool. By default, the Envoy proxies distribute traffic across each service’s load balancing pool using a least requests model, where each request is routed to the host with fewer active requests from a random selection of two hosts from the pool; in this way the most heavily loaded host will not receive requests until it is no more loaded than any other host.

While Istio’s basic service discovery and load balancing gives you a working service mesh, it’s far from all that Istio can do. In many cases you might want more fine-grained control over what happens to your mesh traffic. You might want to direct a particular percentage of traffic to a new version of a service as part of A/B testing, or apply a different load balancing policy to traffic for a particular subset of service instances. You might also want to apply special rules to traffic coming into or out of your mesh, or add an external dependency of your mesh to the service registry. You can do all this and more by adding your own traffic configuration to Istio using Istio’s traffic management API.

Like other Istio configuration, the API is specified using Kubernetes custom resource definitions (CRDs), which you can configure using YAML, as you’ll see in the examples.

- Address multiple application services through a single virtual service. If
your mesh uses Kubernetes, for example, you can configure a virtual service
to handle all services in a specific namespace. Mapping a single
virtual service to multiple “real” services is particularly useful in
facilitating turning a monolithic application into a composite service built
out of distinct microservices without requiring the consumers of the service
to adapt to the transition. Your routing rules can specify “calls to these URIs of
`monolith.com`

go to`microservice A`

”, and so on. You can see how this works in[one of our examples below](/latest/docs/concepts/traffic-management/#more-about-routing-rules). - Configure traffic rules in combination with
[gateways](/latest/docs/concepts/traffic-management/#gateways)to control ingress and egress traffic.

In some cases you also need to configure destination rules to use these features, as these are where you specify your service subsets. Specifying service subsets and other destination-specific policies in a separate object lets you reuse these cleanly between virtual services. You can find out more about destination rules in the next section.

### Virtual service example

The following virtual service routes requests to different versions of a service depending on whether the request comes from a particular user.

```
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
name: reviews
spec:
hosts:
- reviews
http:
- match:
- headers:
end-user:
exact: jason
route:
- destination:
host: reviews
subset: v2
- route:
- destination:
host: reviews
subset: v3
```


#### The hosts field。
            """,
            "expected_chunks": 4
        },
        
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试用例: {case['name']}")
        print('='*50)

        try:
            # 分割文本
            chunks = splitter.split_text(case['text'])
            
            # 打印结果
            print(f"\n1. 基本信息:")
            print(f"分割块数量: {len(chunks)}")
            print(f"预期块数量: {case['expected_chunks']}")
            
            print("\n2. 分割结果:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n块 {i}:")
                print(f"长度: {len(chunk)}")
                print(f"内容: {chunk}")
                
            # 验证分割结果
            assert len(chunks) > 0, "分割结果不能为空"
            for chunk in chunks:
                assert len(chunk) <= 100, f"块大小超过限制: {len(chunk)} > 100"
            
            print("\n✓ 测试通过")
            
        except Exception as e:
            print(f"\n✗ 测试失败: {str(e)}")

    # 测试文档分割
    print("\n测试文档分割功能:")
    try:
        # 创建测试文档
        docs = [
            Document(
                page_content="这是第一个文档的内容。它包含多个句子。这些句子应该被正确分割。",
                metadata={"source": "test1"}
            ),
            Document(
                page_content="This is the content of the second document. It should be kept as is.",
                metadata={"source": "test2"}
            )
        ]
        
        # 分割文档
        split_docs = splitter.split_documents(docs)
        
        print(f"\n分割后文档数量: {len(split_docs)}")
        for i, doc in enumerate(split_docs, 1):
            print(f"\n文档 {i}:")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
        
        print("\n✓ 文档分割测试通过")
        
    except Exception as e:
        print(f"\n✗ 文档分割测试失败: {str(e)}")


if __name__ == "__main__":
    test_splitter()






