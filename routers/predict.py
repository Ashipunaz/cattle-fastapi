import io
import uuid
import numpy as np
from PIL import Image
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from schemas import PredictionResult
from utils.model_loader import get_model, get_active_version
from utils.pdf_generator import build_pdf

router = APIRouter()

CLASS_NAMES = ["fmd", "healthy", "lumpy skin", "mastitis"]
IMG_SIZE    = (224, 224)

DISEASE_INFO = {
    "fmd": {
        "full_name":    "Foot and Mouth Disease",
        "severity":     "Urgent",
        "requires_vet": True,
        "what_you_see": "Blisters on the mouth, tongue, and hooves. The animal may be drooling heavily or struggling to walk.",
        "what_to_do":   "Separate this animal from the rest of the herd right away. Do not move animals off the farm. Call your vet or the nearest livestock office today.",
        "urgency_msg":  "Act today. Every hour matters.",
    },
    "healthy": {
        "full_name":    "No Disease Detected",
        "severity":     "All Clear",
        "requires_vet": False,
        "what_you_see": "The animal shows no visible signs of disease.",
        "what_to_do":   "Your animal looks healthy. Keep up regular check-ups, clean water and feed, and stay on your vaccination schedule.",
        "urgency_msg":  "Continue routine care.",
    },
    "lumpy skin": {
        "full_name":    "Lumpy Skin Disease",
        "severity":     "Urgent",
        "requires_vet": True,
        "what_you_see": "Round raised lumps or nodules appearing across the skin. The animal may have a fever and reduced milk output.",
        "what_to_do":   "Separate the animal immediately. LSD spreads through insect bites — treat the whole herd with insect repellent. Contact your vet to vaccinate at-risk animals.",
        "urgency_msg":  "Isolate today. Protect the herd.",
    },
    "mastitis": {
        "full_name":    "Mastitis",
        "severity":     "Needs Attention",
        "requires_vet": True,
        "what_you_see": "The udder looks swollen or feels warm and painful. Milk may appear watery, lumpy, or discoloured.",
        "what_to_do":   "Contact your vet for antibiotic treatment. Milk affected quarters separately and discard that milk. Wash hands and equipment between animals.",
        "urgency_msg":  "Book a vet visit soon.",
    },
}

# In-memory history store (replace with a database in production)
prediction_history = []


@router.post("/predict", response_model=PredictionResult, summary="Predict disease from image")
async def predict(file: UploadFile = File(..., description="JPG or PNG image of the cattle")):
    """
    Upload an image of cattle and receive a disease prediction.

    Returns the predicted disease, confidence score, severity level,
    and recommended action for the farmer.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are accepted.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 10MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid JPG or PNG.")

    model  = get_model()
    arr    = np.expand_dims(np.array(image, dtype=np.float32), 0)
    preds  = model.predict(arr, verbose=0)[0]
    idx    = int(np.argmax(preds))
    disease = CLASS_NAMES[idx]
    info    = DISEASE_INFO[disease]

    result = {
        "id":              str(uuid.uuid4()),
        "filename":        file.filename,
        "disease":         disease,
        "full_name":       info["full_name"],
        "confidence":      round(float(preds[idx]) * 100, 2),
        "severity":        info["severity"],
        "requires_vet":    info["requires_vet"],
        "what_you_see":    info["what_you_see"],
        "what_to_do":      info["what_to_do"],
        "urgency_msg":     info["urgency_msg"],
        "all_predictions": {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
        "timestamp":       datetime.now().strftime("%d %b %Y %I:%M %p"),
        "model_version":   get_active_version(),
    }

    # Store in history (include raw image bytes for PDF)
    history_entry = dict(result)
    history_entry["image_bytes"] = contents
    prediction_history.append(history_entry)

    return result


@router.get("/predict/{prediction_id}/report", summary="Download PDF report for a prediction")
def download_report(prediction_id: str):
    """
    Download a PDF report for a given prediction ID.
    """
    record = next((r for r in prediction_history if r["id"] == prediction_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    pdf_bytes = build_pdf([record], DISEASE_INFO, CLASS_NAMES)
    filename  = f"CattleReport_{prediction_id[:8]}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )