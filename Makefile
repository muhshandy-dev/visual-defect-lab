.PHONY: install data evaluate report onnx web verify lint fmt test check
install:
	python -m venv .venv && .venv/bin/pip install -e ".[model,dev]"
data:
	.venv/bin/defect-lab data
evaluate:
	for m in classical resnet18 wrn50; do .venv/bin/defect-lab evaluate --method $$m || exit 1; done
report:
	.venv/bin/defect-lab report
onnx:
	.venv/bin/defect-lab export-onnx
web:
	.venv/bin/defect-lab export-web
verify:
	.venv/bin/python tools/verify_published.py
lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
fmt:
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .
test:
	.venv/bin/pytest
check: lint test verify
