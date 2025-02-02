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
	uv sync --all-extras

.PHONY: install_common_ci
install_common_ci:
	@echo "[CI] Installing common Python dependencies"
	uv sync --inexact --frozen --quiet

.PHONY: install_cpu
install_cpu: .install_common .pre-commit
	@echo "Overriding/installing PyTorch (CPU)"
	uv pip install --reinstall \
	"torch>=${TORCH_VERSION}"

.PHONY: install_cuda
install_cuda: .install_common .pre-commit
	@echo "Overriding/installing PyTorch (GPU)"
	uv pip install --reinstall \
	"torch>=${TORCH_VERSION}" --index-url ${CUDA_INDEX_URL}

.PHONY: install_ci
install_ci: install_common_ci	
	@echo "[CI] Overriding/installing PyTorch (CPU)"
	uv pip install --reinstall --quiet \
	"torch>=${TORCH_VERSION}"

.PHONY: install_mamba
install_mamba:
	uv pip install --no-build-isolation \
	"mamba-ssm>=${MAMBA_VERSION}" \
	"causal-conv1d>=${CASUAL_CONV_VERSION}"

.PHONY: check_types
check_types:
	uv run mypy src tests scripts

.PHONY: lint
lint:
	uv run ruff check src tests scripts

.PHONY: format
format:
	uv run ruff format src tests scripts

.PHONY: test
test: test_unit test_integration test_e2e

.PHONY: test_unit
test_unit:
	uv run pytest tests/unit

.PHONY: test_integration
test_integration:
	$(MAKE) test_integration_install
	$(MAKE) test_integration_config

.PHONY: test_integration_install
test_integration_install:
	uv run --project tests/integration/install --reinstall-package mblm \
		--isolated --quiet pytest tests/integration/install

.PHONY: test_integration_config
test_integration_config:
	uv run pytest tests/integration/config

E2E_RUN_TORCH = OMP_NUM_THREADS=1 \
	uv run torchrun --nproc_per_node=2 \
	tests/e2e/trainer/run_trainer.py
E2E_RUN_VALIDATE = uv run tests/e2e/trainer/validate.py
E2E_TEST_ROOT = tests/e2e/trainer

.PHONY: test_e2e
test_e2e:
	@echo "Clearning test output dir $(E2E_TEST_ROOT)/outputs"
	rm -rf $(E2E_TEST_ROOT)/outputs/*
	$(MAKE) test_e2e_grad_acc
	$(MAKE) test_e2e_trainer

.PHONY: test_e2e_grad_acc
test_e2e_grad_acc:
	TEST_ID=grad_acc_1 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/sample-config-grad-acc-1.yaml
	TEST_ID=grad_acc_2 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/sample-config-grad-acc-2.yaml
	$(E2E_RUN_VALIDATE) \
		--check-grad-acc-csv $(E2E_TEST_ROOT)/outputs/my-model_grad_acc_2/loss.csv \
		--check-grad-acc-csv $(E2E_TEST_ROOT)/outputs/my-model_grad_acc_1/loss.csv

.PHONY: test_e2e_trainer
test_e2e_trainer:
	# ---------------- Chain 3 runs of 1 epoch each ----------------
	@echo "Running training for a single epoch"
	TEST_ID=1 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/sample-config-1-epoch.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_1

	@echo "Chaining run 2 to run 1"
	TEST_ID=2 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/outputs/my-model_1/config.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_2

	@echo "Chaining run 3 to run 2"
	TEST_ID=3 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/outputs/my-model_2/config.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_3

	@echo "Asserting on chained training runs 1, 2, 3"
	$(E2E_RUN_VALIDATE) \
		--assert-equal-epochs \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_1/loss.csv \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_2/loss.csv \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_3/loss.csv

	# ------------ Chain 2 runs of more than 1 epoch each ------------
	@echo "Running training for more than 1 epoch"
	TEST_ID=4 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/sample-config-1.5-epoch.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_4

	@echo "Chaining run 5 to run 4"
	TEST_ID=5 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/outputs/my-model_4/config.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_5

	@echo "Asserting on chained training runs 4, 5"
	$(E2E_RUN_VALIDATE) \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_4/loss.csv \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_5/loss.csv

	# ------------ Chain 2 runs of less than 1 epoch each ------------
	@echo "Running training for less than 1 epoch"
	TEST_ID=6 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/sample-config-0.5-epoch.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_6

	@echo "Chaining run 7 to run 6"
	TEST_ID=7 $(E2E_RUN_TORCH) -c $(E2E_TEST_ROOT)/outputs/my-model_6/config.yaml
	$(E2E_RUN_VALIDATE) --check-output $(E2E_TEST_ROOT)/outputs/my-model_7
	
	@echo "Asserting on chained training runs 6, 7"
	$(E2E_RUN_VALIDATE) \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_6/loss.csv \
		--check-chained-csv $(E2E_TEST_ROOT)/outputs/my-model_7/loss.csv

.PHONY: clear_cache
clear_cache:
	find . | grep -E "(__pycache__|\.pyc$$)" | xargs rm -rf