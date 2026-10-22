.PHONY: backend-install backend-run web-install web-dev web-build test test-full openapi health docker-build docker-up docker-down docker-logs clean

BACKEND_DIR=backend
WEB_DIR=web
PY=../venv/Scripts/python.exe

backend-install:
	cd $(BACKEND_DIR) && $(PY) -m pip install -r requirements.txt

backend-run:
	cd $(BACKEND_DIR) && $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

web-install:
	cd $(WEB_DIR) && npm install

web-dev:
	cd $(WEB_DIR) && npm run dev

web-build:
	cd $(WEB_DIR) && npm run build

test:
	cd $(BACKEND_DIR) && $(PY) -m pytest tests/ -q --tb=short

test-full:
	cd $(BACKEND_DIR) && $(PY) -m pytest tests/ -v --tb=long --durations=10

openapi:
	cd $(BACKEND_DIR) && $(PY) scripts/generate_openapi.py

health:
	curl -s http://localhost:8000/api/v1/health | python -m json.tool

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	find $(BACKEND_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(BACKEND_DIR) -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(BACKEND_DIR)/.pytest_cache 2>/dev/null || true
