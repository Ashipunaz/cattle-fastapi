# Cattle Disease Detection — FastAPI

REST API for AI-powered cattle disease detection. Built with FastAPI and TensorFlow.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Upload image → get disease prediction |
| GET | `/api/predict/{id}/report` | Download PDF report |
| GET | `/api/history` | Past predictions |
| GET | `/api/admin/models` | List model versions (admin) |
| POST | `/api/admin/models/upload` | Upload new model (admin) |
| POST | `/api/admin/models/activate` | Switch active model (admin) |
| DELETE | `/api/admin/models/{version}` | Delete a model version (admin) |
| GET | `/api/admin/status` | System status (admin) |

## Setup

```bash
git clone https://github.com/Ashipunaz/cattle-disease-detection.git
cd cattle-fastapi
pip install -r requirements.txt
```

Place your model files in `models/active/`:
- `cattle_final_archi.json`
- `cattle_final.weights.h5`

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Admin Access

Pass `X-Admin-Token` header with all admin requests.

Set token via environment variable:
```bash
export ADMIN_TOKEN=your-secret-token
```

## CI/CD

- Push to `main` → GitHub Actions runs tests → deploys to Render automatically
- Set `RENDER_DEPLOY_HOOK` and `ADMIN_TOKEN` in GitHub repository secrets

## Example — Predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@cow_photo.jpg"
```

Response:
```json
{
  "id": "abc123",
  "disease": "lumpy skin",
  "full_name": "Lumpy Skin Disease",
  "confidence": 94.2,
  "severity": "Urgent",
  "requires_vet": true,
  "urgency_msg": "Isolate today. Protect the herd.",
  "model_version": "v1.0"
}
```