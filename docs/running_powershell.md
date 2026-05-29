# Running on Windows PowerShell

## 1. Enter the project folder

```powershell
cd path\to\primary-reuse-fastapi-public-v2
```

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the public-safe demo

```powershell
$env:PUBLIC_DEMO_MODE="true"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## 5. Stop the server

Press:

```text
Ctrl + C
```

## 6. Clean local environment

To remove the virtual environment:

```powershell
deactivate
Remove-Item -Recurse -Force .\.venv
```
