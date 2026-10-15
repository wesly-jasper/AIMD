# AIMD Deployment Guide

## Quick Start with Docker

```bash
# 1. Clone repository and build containers
docker-compose up --build -d

# 2. Access dashboard
http://localhost:8000
```

## Bare-Metal / Local Deployment

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run build # or npm run dev for live development
```
