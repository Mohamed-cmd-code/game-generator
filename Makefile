.PHONY: install run test demos eval lint

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload

test:
	pytest

demos:
	python -m scripts.run_demos

eval:
	python -m scripts.eval_retrieval

lint:
	ruff check app tests scripts
