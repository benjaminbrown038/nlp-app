.PHONY: 
	init run test fmt lint docker-build docker-run


init:
	bash scripts/setup_env.sh 3.10


run:
	bash scripts/run_dev.sh


test:
	source .venv/bin/activate && pytest -q


fmt:
	source .venv/bin/activate && python -m pip install ruff && ruff check --fix .


lint:
	source .venv/bin/activate && python -m pip install ruff && ruff check .


docker-build:
	docker build -t nlp-app:latest .


docker-run:
	docker run -p 8000:8000 -e ENABLE_TRANSFORMERS=0 nlp-app:latest