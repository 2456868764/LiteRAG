FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"

ARG PKGNAME

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    default-jre \
    libgl1-mesa-glx \
    libjemalloc-dev \
    tesseract-ocr

#RUN useradd -m -s /bin/bash user && \
#    mkdir -p /home/user && \
#    chown -R user /home/user/
#
#USER user

COPY . /home/user/

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi && \
    pip install --no-cache-dir -r /home/user/server/${PKGNAME}/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

#USER root
#RUN mkdir -p /data && chown -R user /data
#USER user

WORKDIR /home/user/server/${PKGNAME}
ENTRYPOINT ["python", "service.py"]
