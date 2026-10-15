.PHONY: backend frontend test docker clean

BACKEND_DIR=backend
FRONTEND_DIR=frontend
PY=D:/clg\ project/inscription\ project/surface_detector/surface_env/Scripts/python.exe

backend-install:
	cd $(BACKEND_DIR) && $(PY) -m pip install -r requirements.txt

backend-run:
	cd $(BACKEND_DIR) && $(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

frontend-install:
	cd $(FRONTEND_DIR) && npm install

frontend-dev:
	cd $(FRONTEND_DIR) && npm run dev

frontend-build:
	cd $(FRONTEND_DIR) && npm run build

test:
	cd $(BACKEND_DIR) && $(PY) -m pytest tests/ -v --tb=short

test-full:
	cd $(BACKEND_DIR) && $(PY) -m pytest tests/ -v --tb=long --durations=10

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
	rm -rf $(BACKEND_DIR)/uploads/* 2>/dev/null || true
	rm -rf $(BACKEND_DIR)/processed/frames/* 2>/dev/null || true
