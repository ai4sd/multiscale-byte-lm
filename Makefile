TORCH_VERSION = 2.4.1
CUDA_INDEX_URL = https://download.pytorch.org/whl/cu124

MAMBA_VERSION = 2.2.2
CASUAL_CONV_VERSION = 1.4.0

.DEFAULT_GOAL := all

.PHONY: all
all: format lint check_types test 

.PHONY: .pre-commit
.pre-commit:
	@echo "Installing pre-commit hook"
	uv run pre-commit install

.PHONY: .install_common	
.install_common:
	@echo "Installing common Python dependencies"
	uv sync --inexact --frozen

.PHONY: install_common_ci	
install_common_ci:
	@echo "[CI] Installing common Python dependencies"
	uv sync --inexact --frozen --no-cache --quiet

.PHONY: install_cpu
install_cpu: .install_common .pre-commit
	@echo "Overriding/installing PyTorch (CPU)"
	uv pip install --reinstall \
	torch==${TORCH_VERSION}

.PHONY: install_cuda
install_cuda: .install_common .pre-commit
	@echo "Overriding/installing PyTorch (GPU)"
	uv pip install --reinstall \
	torch==${TORCH_VERSION} --index-url ${CUDA_INDEX_URL}

.PHONY: install_ci
install_ci: install_common_ci	
	@echo "[CI] Overriding/installing PyTorch (CPU)"
	uv pip install --reinstall --no-cache --quiet \
	torch==${TORCH_VERSION}

.PHONY: install_mamba
install_mamba:
	uv pip install --no-build-isolation \
	mamba-ssm==${MAMBA_VERSION} \
	causal-conv1d==${CASUAL_CONV_VERSION}
	
.PHONY: check_types
check_types:
	uv run mypy src

.PHONY: lint
lint:
	uv run ruff check src

.PHONY: format
format:
	uv run ruff format src

.PHONY: clear_cache
clear_cache:
	find . | grep -E "(__pycache__|\.pyc$$)" | xargs rm -rf