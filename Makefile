.PHONY: setup dev test lint train eval

setup:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

train:
	python -m orion.train --config configs/golden.yaml

eval:
	python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt

