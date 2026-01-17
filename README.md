# RecycleLens

REST API that classifies objects in photos as recyclable or non recyclable.

## Requirements

- Python 3.8+
- (runtime) fastapi, uvicorn, ultralytics
- (image handling in code/tests) pillow, numpy
- (testing) pytest, requests, httpx, python-multipart

You can install runtime requirements with:

```bash
python -m venv .venv
# Activate the venv (Windows examples)
# Command Prompt:
.venv\Scripts\activate
# PowerShell:
.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate

pip install -r requirements.txt
# Install extras for development/testing
pip install -r requirements-dev.txt
```

You can run the development server with:
```bash
# Run Dev Server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Pytorch model must be installed manually.