#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting crawler in background..."
nohup python -c "
from rag.tasks.url_crawler import URLCrawler
crawler = URLCrawler()
crawler.run()
" > /tmp/literag_crawler.log 2>&1 &

# 保存进程ID
echo $! > /tmp/literag_crawler.pid
echo "Crawler started with PID: $!" 