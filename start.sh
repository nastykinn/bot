#!/bin/bash

# Запуск FastAPI сервера с uvicorn
exec uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
