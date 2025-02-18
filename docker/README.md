# huggingface

```shell
sudo ubuntu-drivers autoinstall
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
mkdir /data
huggingface-cli download --resume-download BAAI/bge-base-en-v1.5   --local-dir data/models--BAAI--bge-base-en-v1.5
huggingface-cli download --resume-download Intel/neural-chat-7b-v3-3   --local-dir data/models--Intel--neural-chat-7b-v3-3
huggingface-cli download --resume-download BAAI/bge-reranker-base   --local-dir data/models--BAAI--bge-reranker-base
huggingface-cli download --resume-download maidalun1020/bce-embedding-base_v1   --local-dir data/models--maidalun1020--bce-embedding-base_v1
huggingface-cli download --resume-download maidalun1020/bce-reranker-base_v1    --local-dir data/models--maidalun1020--bce-reranker-base_v1
huggingface-cli download --resume-download Qwen/Qwen2-1.5B    --local-dir data/models--Qwen--Qwen2-1.5B
huggingface-cli download --resume-download Qwen/Qwen2-7B    --local-dir data/models--Qwen--Qwen2-7B


```

#
```shell
https://help.aliyun.com/zh/egs/user-guide/install-a-gpu-driver-on-a-gpu-accelerated-compute-optimized-linux-instance?spm=a2c4g.11186623.0.0.20df4699AmYHrh
sudo dpkg -i package_name.deb
lspci | grep -i nvidia
nvidia-smi
lsmod | grep nouveau
sudo apt-get update
sudo apt upgrade -y
```

# .env

```shell
EMBEDDING_MODEL_ID=BAAI/bge-base-en-v1.5
RERANK_MODEL_ID=BAAI/bge-reranker-base
LLM_MODEL_ID=Intel/neural-chat-7b-v3-3
INDEX_NAME=rag-redis
no_proxy=localhost,127.0.0.1,172.22.105.215
HUGGINGFACEHUB_API_TOKEN=xxxxxxx
```

# docker installation

```shell
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt update
sudo apt upgrade -y
sudo apt-get install ca-certificates curl gnupg lsb-release software-properties-common apt-transport-https
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get install docker-ce docker-ce-cli containerd.io
systemctl start docker
docker info
```

# conda

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

conda create -n literag python=3.11
conda activate literag
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:/root/000016
python service.py

```

# ssh
```shell
ssh-keygen

```
# git

```shell
git clone git@atomgit.com:opea/000016.git
```

# embedding

```shelll
curl -v http://127.0.0.1:6006/embed  \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
    
    
curl 127.0.0.1:8808/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'    
```



# build images

```shell

GIT_VERSION=v0.0.7 make image-dataprep
GIT_VERSION=v0.0.7 make push-image-dataprep

GIT_VERSION=v0.0.7 make image-retriever
GIT_VERSION=v0.0.7 make push-image-retriever

GIT_VERSION=v0.0.7 make image-chatqna
GIT_VERSION=v0.0.7 make push-image-chatqna
```

# test
export host_ip=127.0.0.1
```shell
curl -X POST  http://${host_ip}:6010/v1/knowledge/list

curl -X POST -F "knowledge_name=istio" http://${host_ip}:6010/v1/knowledge/create 

wget https://raw.githubusercontent.com/opea-project/GenAIComps/main/comps/retrievers/redis/data/nke-10k-2023.pdf
# upload pdf file with dataprep
curl -X POST "http://${host_ip}:6010/v1/knowledge/uploaded_docs" \
    -H "Content-Type: multipart/form-data" \
    -F "knowledge_name=istio" \
    -F "files=@./nke-10k-2023.pdf"    
   
    
curl -X POST "http://${host_ip}:7000/v1/retrieval" \
    -H "Content-Type: application/json" \
    -d '{
  "text": "Istio 核心特性?",
  "embedding": [[0.123, 0.456, 0.789], [0.234, 0.567, 0.890]],
  "search_type": "mmr",
  "k": 4,
  "distance_threshold": 0.5,
  "fetch_k": 20,
  "lambda_mult": 0.5,
  "score_threshold": 0.2,
  "knowledge_name": "istio"
}'
    
```

    text: Union[str, List[str]]
    embedding: Union[conlist(float, min_length=0), List[conlist(float, min_length=0)]]
    search_type: str = "similarity"
    k: int = 4
    distance_threshold: Optional[float] = None
    fetch_k: int = 20
    lambda_mult: float = 0.5
    score_threshold: float = 0.2
    knowledge_name: Optional[str] = None
    constraints: Optional[Union[Dict[str, Any], List[Dict[str, Any]], None]] = None



```shell
export host_ip=172.22.105.215
curl http://${host_ip}:8888/v1/chatqna \
    -H "Content-Type: application/json" \
    -d '{
        "messages": "What is the revenue of Nike in 2023?"
    }'
    
wget https://raw.githubusercontent.com/opea-project/GenAIComps/main/comps/retrievers/redis/data/nke-10k-2023.pdf
# upload pdf file with dataprep
curl -X POST "http://${host_ip}:6007/v1/dataprep" \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./nke-10k-2023.pdf"    
    
```

