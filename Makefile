.PHONY: setup dev test lint format train eval smoke

setup:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

test:
	pytest -q

smoke:
	SMOKE_TEST=true SMOKE_STEPS=20 python -m orion.train --config configs/golden.yaml

train:
	python -m orion.train --config configs/golden.yaml

eval:
	python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt

