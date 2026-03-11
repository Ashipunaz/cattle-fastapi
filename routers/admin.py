import os
import shutil
from fastapi import APIRouter, File, UploadFile, Header, HTTPException, Form
from typing import List, Optional
from schemas import ModelVersion, APIResponse
from utils.model_loader import list_versions, activate_version, get_active_version, VERSION_DIR

router = APIRouter()

# Admin password — in production load from environment variable
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "cattle-admin-2024")

ARCH_FILE    = "cattle_final_archi.json"
WEIGHTS_FILE = "cattle_final.weights.h5"


def verify_admin(x_admin_token: Optional[str]):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token. Pass X-Admin-Token header.")


@router.get("/models", response_model=List[ModelVersion], summary="List all model versions")
def get_models(x_admin_token: Optional[str] = Header(None)):
    """
    Returns all uploaded model versions and which one is currently active.
    Requires admin token in X-Admin-Token header.
    """
    verify_admin(x_admin_token)
    return list_versions()


@router.post("/models/activate", response_model=APIResponse, summary="Switch active model version")
def activate_model(version: str = Form(...), x_admin_token: Optional[str] = Header(None)):
    """
    Switch the active model to a specific version.
    The API will immediately start using the new model for predictions.
    Requires admin token in X-Admin-Token header.
    """
    verify_admin(x_admin_token)
    try:
        activated = activate_version(version)
        return {"success": True, "message": f"Model version '{activated}' is now active.", "data": {"version": activated}}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {e}")


@router.post("/models/upload", response_model=APIResponse, summary="Upload a new model version")
async def upload_model(
    version:     str        = Form(..., description="Version name e.g. v2.0"),
    arch_file:   UploadFile = File(..., description="Architecture JSON file"),
    weights_file: UploadFile = File(..., description="Weights .h5 file"),
    x_admin_token: Optional[str] = Header(None),
):
    """
    Upload a new model version (architecture JSON + weights H5).
    The version is saved but NOT automatically activated — use /activate to switch.
    Requires admin token in X-Admin-Token header.
    """
    verify_admin(x_admin_token)

    # Sanitise version name
    version = version.strip().replace(" ", "_")
    if not version:
        raise HTTPException(status_code=400, detail="Version name cannot be empty.")

    version_path = os.path.join(VERSION_DIR, version)
    if os.path.exists(version_path):
        raise HTTPException(status_code=400, detail=f"Version '{version}' already exists.")

    os.makedirs(version_path, exist_ok=True)

    try:
        # Save architecture file
        arch_bytes = await arch_file.read()
        with open(os.path.join(version_path, ARCH_FILE), "wb") as f:
            f.write(arch_bytes)

        # Save weights file
        weights_bytes = await weights_file.read()
        with open(os.path.join(version_path, WEIGHTS_FILE), "wb") as f:
            f.write(weights_bytes)

    except Exception as e:
        shutil.rmtree(version_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "success": True,
        "message": f"Model version '{version}' uploaded successfully. Use /activate to switch to it.",
        "data": {"version": version}
    }


@router.delete("/models/{version}", response_model=APIResponse, summary="Delete a model version")
def delete_model(version: str, x_admin_token: Optional[str] = Header(None)):
    """
    Delete a stored model version. Cannot delete the currently active version.
    Requires admin token in X-Admin-Token header.
    """
    verify_admin(x_admin_token)

    if version == get_active_version():
        raise HTTPException(status_code=400, detail="Cannot delete the currently active model version.")

    version_path = os.path.join(VERSION_DIR, version)
    if not os.path.exists(version_path):
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    shutil.rmtree(version_path)
    return {"success": True, "message": f"Version '{version}' deleted."}


@router.get("/status", summary="Admin status overview")
def admin_status(x_admin_token: Optional[str] = Header(None)):
    """
    Returns current system status — active model, total versions, history count.
    """
    verify_admin(x_admin_token)
    from routers.predict import prediction_history
    return {
        "active_model":    get_active_version(),
        "total_versions":  len(list_versions()),
        "total_predictions": len(prediction_history),
    }