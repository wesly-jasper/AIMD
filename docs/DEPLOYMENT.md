# Deployment

## Local

```bash
# Backend (from backend/, repository-root venv)
../venv/Scripts/python.exe -m pip install -r requirements.txt
../venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
# Migrations run on startup (alembic upgrade head); no separate step.

# Web app (from web/)
npm install
npm run dev        # http://localhost:5173, proxies /api to 127.0.0.1:8000
```

Settings come from `backend/.env` (copy `.env.example`). The ones that matter:

- `AIMD_JWT_SECRET`: 32+ characters. Required outside `APP_ENV=development`.
- `AIMD_BOOTSTRAP_ADMIN_USERNAME` / `AIMD_BOOTSTRAP_ADMIN_PASSWORD`: creates the
  first admin once. Everyone else signs up in the app.
- `DAILY_UPLOAD_LIMIT`: uploads per account per 24 h (default 50).

## Docker

```bash
docker compose up --build -d     # API on http://localhost:8000
```

The image builds `web/dist` but the API does **not** serve it yet. Serve
`web/dist` from a static host or reverse proxy that forwards `/api` to port 8000.
