import os
import json
import threading
import logging
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build path relative to main.py (project root)
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

# Debug startup
logger.info("=" * 50)
logger.info("MODEL LOADER INITIALIZATION")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"ACTIVE_DIR: {ACTIVE_DIR}")
logger.info(f"TensorFlow version: {tf.__version__}")

# Check active directory
if os.path.exists(ACTIVE_DIR):
    files = os.listdir(ACTIVE_DIR)
    logger.info(f"Files in active directory: {files}")
    
    # Detailed check for .keras file
    keras_path = os.path.join(ACTIVE_DIR, KERAS_FILE)
    if os.path.exists(keras_path):
        file_size = os.path.getsize(keras_path)
        logger.info(f"Found {KERAS_FILE} - Size: {file_size} bytes")
        
        # Check if it's a valid zip file
        try:
            import zipfile
            with zipfile.ZipFile(keras_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                logger.info(f"Valid zip file. Contents: {file_list[:5]}...")
        except Exception as e:
            logger.error(f"File exists but is NOT a valid zip file: {e}")
    else:
        logger.error(f"{KERAS_FILE} NOT FOUND!")
else:
    logger.error(f"ACTIVE_DIR does not exist: {ACTIVE_DIR}")
    os.makedirs(ACTIVE_DIR, exist_ok=True)
logger.info("=" * 50)


def get_active_version() -> str:
    """Get the current active model version."""
    version_file = os.path.join(ACTIVE_DIR, "version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return "v1.0"


def load_with_custom_objects(model_path):
    """Load model with custom objects handling."""
    try:
        # Define custom objects if needed
        custom_objects = {}
        
        # Try loading with different options
        try:
            # Method 1: Standard load
            model = tf.keras.models.load_model(model_path)
            logger.info("Method 1: Standard load successful")
            return model
        except Exception as e1:
            logger.warning(f"Method 1 failed: {e1}")
            
            try:
                # Method 2: Load without compilation
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Method 2: Load with compile=False successful")
                
                # Compile with default settings
                model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )
                return model
            except Exception as e2:
                logger.warning(f"Method 2 failed: {e2}")
                
                try:
                    # Method 3: Load with custom objects
                    model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    logger.info("Method 3: Load with custom objects successful")
                    
                    model.compile(
                        optimizer=Adam(learning_rate=1e-5),
                        loss="categorical_crossentropy",
                        metrics=["accuracy"]
                    )
                    return model
                except Exception as e3:
                    logger.error(f"All loading methods failed. Last error: {e3}")
                    raise
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def create_model_from_weights():
    """Create model from architecture JSON and weights file."""
    arch_path = os.path.join(ACTIVE_DIR, ARCH_FILE)
    weights_path = os.path.join(ACTIVE_DIR, WEIGHTS_FILE)
    
    if not (os.path.exists(arch_path) and os.path.exists(weights_path)):
        raise FileNotFoundError(f"Missing JSON or weights files")
    
    logger.info("Creating model from JSON architecture and weights")
    
    # Load architecture
    with open(arch_path, 'r') as f:
        config = json.load(f)
    
    # Fix configuration
    def fix_config(obj):
        if isinstance(obj, dict):
            if obj.get('class_name') == 'InputLayer':
                cfg = obj.get('config', {})
                if 'batch_input_shape' in cfg and 'shape' not in cfg:
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
    
    # Create model
    try:
        model = tf.keras.Model.from_config(config)
        logger.info("Model created from config")
    except Exception as e:
        logger.error(f"Failed to create model from config: {e}")
        raise
    
    # Load weights
    try:
        model.load_weights(weights_path)
        logger.info("Weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def _load_from_disk():
    """Load the model from disk with multiple fallback strategies."""
    global _model, _model_version
    
    logger.info("=" * 50)
    logger.info("LOADING MODEL FROM DISK")
    
    # Ensure directory exists
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    
    keras_path = os.path.join(ACTIVE_DIR, KERAS_FILE)
    arch_path = os.path.join(ACTIVE_DIR, ARCH_FILE)
    weights_path = os.path.join(ACTIVE_DIR, WEIGHTS_FILE)
    
    # Log file status
    logger.info(f"Checking files in {ACTIVE_DIR}:")
    logger.info(f"  - {KERAS_FILE}: {os.path.exists(keras_path)}")
    logger.info(f"  - {ARCH_FILE}: {os.path.exists(arch_path)}")
    logger.info(f"  - {WEIGHTS_FILE}: {os.path.exists(weights_path)}")
    
    model = None
    
    # Strategy 1: Try loading .keras file
    if os.path.exists(keras_path):
        try:
            logger.info(f"Strategy 1: Loading .keras file: {keras_path}")
            model = load_with_custom_objects(keras_path)
        except Exception as e:
            logger.error(f"Strategy 1 failed: {e}")
            model = None
    
    # Strategy 2: Create from JSON + weights
    if model is None and os.path.exists(arch_path) and os.path.exists(weights_path):
        try:
            logger.info("Strategy 2: Creating model from JSON + weights")
            model = create_model_from_weights()
            
            # Save as .keras for future loads
            try:
                model.save(keras_path)
                logger.info(f"Saved model as {keras_path} for future loads")
            except Exception as save_err:
                logger.warning(f"Could not save .keras file: {save_err}")
                
        except Exception as e:
            logger.error(f"Strategy 2 failed: {e}")
            model = None
    
    # Strategy 3: Try loading with safe mode
    if model is None and os.path.exists(keras_path):
        try:
            logger.info("Strategy 3: Attempting safe mode loading")
            import h5py
            
            # Try to open as H5 file
            with h5py.File(keras_path, 'r') as f:
                logger.info(f"File opens as H5. Keys: {list(f.keys())}")
            
            # Try loading with experimental flags
            model = tf.keras.models.load_model(
                keras_path,
                compile=False,
                safe_mode=False
            )
            logger.info("Strategy 3 successful")
            
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        except Exception as e:
            logger.error(f"Strategy 3 failed: {e}")
            model = None
    
    # If still no model, raise error
    if model is None:
        error_msg = f"Failed to load model from {ACTIVE_DIR}. "
        if os.path.exists(ACTIVE_DIR):
            files = os.listdir(ACTIVE_DIR)
            file_sizes = {f: os.path.getsize(os.path.join(ACTIVE_DIR, f)) for f in files}
            error_msg += f"Files and sizes: {file_sizes}"
        else:
            error_msg += "Directory does not exist!"
        
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    _model = model
    _model_version = get_active_version()
    
    logger.info(f"Model loaded successfully. Version: {_model_version}")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
    logger.info("=" * 50)
    
    return model


def get_model():
    """Get the loaded model (thread-safe)."""
    global _model
    with _model_lock:
        if _model is None:
            _load_from_disk()
    return _model


def reload_model():
    """Force reload the model from disk."""
    global _model
    with _model_lock:
        logger.info("Reloading model...")
        _model = None
        _load_from_disk()
    return _model_version


def list_versions():
    """List all available model versions."""
    os.makedirs(VERSION_DIR, exist_ok=True)
    versions = []
    active = get_active_version()
    
    for folder in sorted(os.listdir(VERSION_DIR)):
        folder_path = os.path.join(VERSION_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        weights = os.path.join(folder_path, WEIGHTS_FILE)
        size_kb = os.path.getsize(weights) / 1024 if os.path.exists(weights) else 0
        
        import datetime
        mtime = os.path.getmtime(folder_path)
        
        versions.append({
            "version": folder,
            "filename": WEIGHTS_FILE,
            "uploaded_at": datetime.datetime.fromtimestamp(mtime).strftime("%d %b %Y %H:%M"),
            "is_active": folder == active,
            "size_kb": round(size_kb, 1),
        })
    
    return versions


def activate_version(version: str):
    """Activate a specific model version."""
    import shutil
    
    version_path = os.path.join(VERSION_DIR, version)
    if not os.path.isdir(version_path):
        raise ValueError(f"Version '{version}' not found")
    
    logger.info(f"Activating version: {version}")
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    
    # Copy files
    for fname in [ARCH_FILE, WEIGHTS_FILE, KERAS_FILE]:
        src = os.path.join(version_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ACTIVE_DIR, fname))
            logger.info(f"Copied {fname}")
    
    # Clean up
    for fname in ["model.keras", KERAS_FILE]:
        p = os.path.join(ACTIVE_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
    
    # Update version
    with open(os.path.join(ACTIVE_DIR, "version.txt"), "w") as f:
        f.write(version)
    
    reload_model()
    return version