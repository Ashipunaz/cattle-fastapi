import os
import json
import threading
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

MODEL_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ACTIVE_DIR  = os.path.join(MODEL_DIR, "active")
VERSION_DIR = os.path.join(MODEL_DIR, "versions")

ARCH_FILE    = "cattle_final_archi.json"
WEIGHTS_FILE = "cattle_final.weights.h5"

# Thread-safe model cache
_model       = None
_model_lock  = threading.Lock()
_model_version = "unknown"


def get_active_version() -> str:
    version_file = os.path.join(ACTIVE_DIR, "version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return "v1.0"


def _load_from_disk():
    global _model, _model_version

    arch_path    = os.path.join(ACTIVE_DIR, ARCH_FILE)
    weights_path = os.path.join(ACTIVE_DIR, WEIGHTS_FILE)
    keras_path   = os.path.join(ACTIVE_DIR, "model.keras")

    # Try cached .keras first
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path)
    elif os.path.exists(arch_path) and os.path.exists(weights_path):
        with open(arch_path) as f:
            config = json.load(f)

        def fix_config(obj):
            if isinstance(obj, dict):
                for key in ["groups", "batch_input_shape"]:
                    obj.pop(key, None)
                for v in obj.values():
                    fix_config(v)
            elif isinstance(obj, list):
                for item in obj:
                    fix_config(item)

        fix_config(config)
        model = tf.keras.Model.from_config(config)
        model.load_weights(weights_path)
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        try:
            model.save(keras_path)
        except Exception:
            pass
    else:
        raise FileNotFoundError(
            f"No model files found in {ACTIVE_DIR}. "
            f"Place {ARCH_FILE} and {WEIGHTS_FILE} there."
        )

    _model         = model
    _model_version = get_active_version()
    return model


def get_model():
    """Return the cached model, loading it if necessary."""
    global _model
    with _model_lock:
        if _model is None:
            _load_from_disk()
    return _model


def reload_model():
    """Force reload — called after a model version switch."""
    global _model
    with _model_lock:
        _model = None
        _load_from_disk()
    return _model_version


def list_versions():
    """Return all uploaded model versions."""
    os.makedirs(VERSION_DIR, exist_ok=True)
    versions = []
    active   = get_active_version()

    for folder in sorted(os.listdir(VERSION_DIR)):
        folder_path = os.path.join(VERSION_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        weights = os.path.join(folder_path, WEIGHTS_FILE)
        size_kb = os.path.getsize(weights) / 1024 if os.path.exists(weights) else 0
        stat    = os.stat(folder_path)
        import datetime
        versions.append({
            "version":     folder,
            "filename":    WEIGHTS_FILE,
            "uploaded_at": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M"),
            "is_active":   folder == active,
            "size_kb":     round(size_kb, 1),
        })
    return versions


def activate_version(version: str):
    """Switch the active model to a given version."""
    version_path = os.path.join(VERSION_DIR, version)
    if not os.path.isdir(version_path):
        raise ValueError(f"Version '{version}' not found.")

    # Copy version files to active directory
    import shutil
    os.makedirs(ACTIVE_DIR, exist_ok=True)

    for fname in [ARCH_FILE, WEIGHTS_FILE]:
        src = os.path.join(version_path, fname)
        dst = os.path.join(ACTIVE_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Remove cached .keras so it rebuilds
    keras_path = os.path.join(ACTIVE_DIR, "model.keras")
    if os.path.exists(keras_path):
        os.remove(keras_path)

    # Write active version marker
    with open(os.path.join(ACTIVE_DIR, "version.txt"), "w") as f:
        f.write(version)

    reload_model()
    return version