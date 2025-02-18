# Git information
GIT_VERSION ?= $(shell git describe --tags --always)
GIT_COMMIT_HASH ?= $(shell git rev-parse HEAD)
GIT_TREESTATE = "clean"
GIT_DIFF = $(shell git diff --quiet >/dev/null 2>&1; if [ $$? -eq 1 ]; then echo "1"; fi)
ifeq ($(GIT_DIFF), 1)
    GIT_TREESTATE = "dirty"
endif
BUILDDATE = $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')


# Images management
REGISTRY ?= registry.cn-hangzhou.aliyuncs.com
REGISTRY_NAMESPACE?= 2456868764
REGISTRY_USER_NAME?=""
REGISTRY_PASSWORD?=""

# Image URL to use all building/pushing image targets
DATAPREP_IMG ?= "${REGISTRY}/${REGISTRY_NAMESPACE}/dataprep:${GIT_VERSION}"

RETRIEVER_IMG ?= "${REGISTRY}/${REGISTRY_NAMESPACE}/retriever:${GIT_VERSION}"

EMBEDDING_IMG ?= "${REGISTRY}/${REGISTRY_NAMESPACE}/embedding:${GIT_VERSION}"

RERANK_IMG ?= "${REGISTRY}/${REGISTRY_NAMESPACE}/rerank:${GIT_VERSION}"

CHATQNA_IMG ?= "${REGISTRY}/${REGISTRY_NAMESPACE}/chatqna:${GIT_VERSION}"

## docker buildx support platform
PLATFORMS ?= linux/arm64,linux/amd64


# Setting SHELL to bash allows bash commands to be executed by recipes.
# This is a requirement for 'setup-envtest.sh' in the test target.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk commands is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: image-dataprep
image-dataprep: ## Build docker image with the dataprep.
	docker build --build-arg PKGNAME=dataprep -t ${DATAPREP_IMG} .

.PHONY: image-retriever
image-retriever: ## Build docker image with the retriever.
	docker build --build-arg PKGNAME=retriever -t ${RETRIEVER_IMG} .

# .PHONY: image-embedding
# image-embedding: ## Build docker image with the embedding.
# 	docker build --build-arg PKGNAME=embedding -t ${EMBEDDING_IMG} .
#
# .PHONY: image-rerank
# image-rerank: ## Build docker image with the rerank.
# 	docker build --build-arg PKGNAME=rerank -t ${RERANK_IMG} .
#

.PHONY: image-chatqna
image-chatqna: ## Build docker image with the chatqna.
	docker build --build-arg PKGNAME=chatqna -t ${CHATQNA_IMG} .

.PHONY: push-image-dataprep
push-image-dataprep: ## Push dataprep images.
ifneq ($(REGISTRY_USER_NAME), "")
	docker login -u $(REGISTRY_USER_NAME) -p $(REGISTRY_PASSWORD) ${REGISTRY}
endif
	docker push ${DATAPREP_IMG}


.PHONY: push-image-retriever
push-image-retriever: ## Push retriever images.
ifneq ($(REGISTRY_USER_NAME), "")
	docker login -u $(REGISTRY_USER_NAME) -p $(REGISTRY_PASSWORD) ${REGISTRY}
endif
	docker push ${RETRIEVER_IMG}

# .PHONY: push-image-embedding
# push-image-embedding: ## Push embedding images.
# ifneq ($(REGISTRY_USER_NAME), "")
# 	docker login -u $(REGISTRY_USER_NAME) -p $(REGISTRY_PASSWORD) ${REGISTRY}
# endif
# 	docker push ${EMBEDDING_IMG}
#
# .PHONY: push-image-rerank
# push-image-rerank: ## Push rerank images.
# ifneq ($(REGISTRY_USER_NAME), "")
# 	docker login -u $(REGISTRY_USER_NAME) -p $(REGISTRY_PASSWORD) ${REGISTRY}
# endif
# 	docker push ${RERANK_IMG}

.PHONY: push-image-chatqna
push-image-chatqna: ## Push chatqna images.
ifneq ($(REGISTRY_USER_NAME), "")
	docker login -u $(REGISTRY_USER_NAME) -p $(REGISTRY_PASSWORD) ${REGISTRY}
endif
	docker push ${CHATQNA_IMG}
