import os
import json
import threading
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Build path relative to main.py (project root), not this file
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
ACTIVE_DIR  = os.path.join(MODEL_DIR, "active")
VERSION_DIR = os.path.join(MODEL_DIR, "versions")

ARCH_FILE    = "cattle_final_archi.json"
WEIGHTS_FILE = "cattle_final.weights.h5"
KERAS_FILE   = "cattle_final.keras"

_model         = None
_model_lock    = threading.Lock()
_model_version = "unknown"

# Debug — print paths on startup so we can see them in Render logs
print(f"[model_loader] BASE_DIR:   {BASE_DIR}")
print(f"[model_loader] ACTIVE_DIR: {ACTIVE_DIR}")
print(f"[model_loader] Files in active: {os.listdir(ACTIVE_DIR) if os.path.exists(ACTIVE_DIR) else 'FOLDER NOT FOUND'}")


def get_active_version() -> str:
    version_file = os.path.join(ACTIVE_DIR, "version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return "v1.0"


def _load_from_disk():
    global _model, _model_version

    keras_candidates = [
        os.path.join(ACTIVE_DIR, KERAS_FILE),
        os.path.join(ACTIVE_DIR, "model.keras"),
    ]
    arch_path    = os.path.join(ACTIVE_DIR, ARCH_FILE)
    weights_path = os.path.join(ACTIVE_DIR, WEIGHTS_FILE)
    keras_path   = next((p for p in keras_candidates if os.path.exists(p)), None)

    print(f"[model_loader] keras_path found: {keras_path}")
    print(f"[model_loader] arch_path exists: {os.path.exists(arch_path)}")
    print(f"[model_loader] weights_path exists: {os.path.exists(weights_path)}")

    if keras_path:
        print(f"[model_loader] Loading from .keras: {keras_path}")
        model = tf.keras.models.load_model(keras_path)
    elif os.path.exists(arch_path) and os.path.exists(weights_path):
        print("[model_loader] Loading from JSON + weights")
        with open(arch_path) as f:
            config = json.load(f)

        def fix_config(obj):
            if isinstance(obj, dict):
                if obj.get('class_name') == 'InputLayer':
                    cfg = obj.get('config', {})
                    if 'shape' not in cfg and 'batch_input_shape' in cfg:
                        cfg['shape'] = cfg['batch_input_shape'][1:]
                    elif 'shape' not in cfg:
                        cfg['shape'] = (224, 224, 3)
                for key in ['groups', 'batch_input_shape']:
                    obj.pop(key, None)
                for v in obj.values():
                    fix_config(v)
            elif isinstance(obj, list):
                for item in obj:
                    fix_config(item)

        fix_config(config)
        model = tf.keras.Model.from_config(config)
        model.load_weights(weights_path)
        model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        try:
            model.save(os.path.join(ACTIVE_DIR, "model.keras"))
        except Exception:
            pass
    else:
        raise FileNotFoundError(
            f"No model files found in {ACTIVE_DIR}. "
            f"Contents: {os.listdir(ACTIVE_DIR) if os.path.exists(ACTIVE_DIR) else 'FOLDER MISSING'}"
        )

    _model         = model
    _model_version = get_active_version()
    print(f"[model_loader] Model loaded successfully. Version: {_model_version}")
    return model


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _load_from_disk()
    return _model


def reload_model():
    global _model
    with _model_lock:
        _model = None
        _load_from_disk()
    return _model_version


def list_versions():
    os.makedirs(VERSION_DIR, exist_ok=True)
    versions = []
    active   = get_active_version()
    for folder in sorted(os.listdir(VERSION_DIR)):
        folder_path = os.path.join(VERSION_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        weights  = os.path.join(folder_path, WEIGHTS_FILE)
        size_kb  = os.path.getsize(weights) / 1024 if os.path.exists(weights) else 0
        import datetime
        versions.append({
            "version":     folder,
            "filename":    WEIGHTS_FILE,
            "uploaded_at": datetime.datetime.fromtimestamp(os.stat(folder_path).st_mtime).strftime("%d %b %Y %H:%M"),
            "is_active":   folder == active,
            "size_kb":     round(size_kb, 1),
        })
    return versions


def activate_version(version: str):
    version_path = os.path.join(VERSION_DIR, version)
    if not os.path.isdir(version_path):
        raise ValueError(f"Version '{version}' not found.")
    import shutil
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    for fname in [ARCH_FILE, WEIGHTS_FILE, KERAS_FILE]:
        src = os.path.join(version_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ACTIVE_DIR, fname))
    for fname in ["model.keras", KERAS_FILE]:
        p = os.path.join(ACTIVE_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
    with open(os.path.join(ACTIVE_DIR, "version.txt"), "w") as f:
        f.write(version)
    reload_model()
    return version