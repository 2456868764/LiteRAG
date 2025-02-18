#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
#python rag/tasks/url_crawler.py 
python rag/module/indexing/loader/web_loader.py