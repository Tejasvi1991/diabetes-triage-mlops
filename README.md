# Virtual Diabetes Clinic Triage

Educational regression API using scikit-learn's Diabetes dataset; not a clinical risk model.

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m training.train --version v0.1
ruff check .
pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/health` or `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

The prediction endpoint returns `{"prediction": <number>}`. Missing fields, incorrect types, and unknown fields return HTTP 422 with FastAPI/Pydantic's JSON `detail` array containing field location, error type and description.

## Docker

```bash
docker build --build-arg MODEL_VERSION=v0.1 -t diabetes-triage:v0.1 .
docker run --rm -p 80:80 diabetes-triage:v0.1
```

## Version 2

Run `python -m training.train --version v0.2`, compare its RMSE with v0.1 on the same split, and update CHANGELOG.md. Build with `--build-arg MODEL_VERSION=v0.2`. Do not overwrite the v0.1 model/image before preserving its release.

## Release

Push a tag `v0.1` or `v0.2` only when ready. The release workflow trains that version, tests, builds, smoke-tests, pushes to GHCR, and creates a GitHub Release. Make GHCR package visibility **public** in GitHub package settings; repository visibility alone may not suffice.
