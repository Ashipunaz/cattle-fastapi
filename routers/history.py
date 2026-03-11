from fastapi import APIRouter, Query
from typing import List
from schemas import HistoryItem
from routers.predict import prediction_history

router = APIRouter()


@router.get("/history", response_model=List[HistoryItem], summary="Get prediction history")
def get_history(limit: int = Query(50, description="Max number of records to return")):
    """
    Returns a list of past predictions, most recent first.
    """
    records = prediction_history[-limit:][::-1]
    return [
        {
            "id":            r["id"],
            "filename":      r["filename"],
            "disease":       r["disease"],
            "full_name":     r["full_name"],
            "confidence":    r["confidence"],
            "severity":      r["severity"],
            "timestamp":     r["timestamp"],
            "model_version": r["model_version"],
        }
        for r in records
    ]


@router.delete("/history", summary="Clear all prediction history")
def clear_history():
    """
    Clears all stored prediction history. Admin use only.
    """
    prediction_history.clear()
    return {"success": True, "message": "History cleared."}