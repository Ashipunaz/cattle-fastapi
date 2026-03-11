import os
import json
import threading
import logging
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Debug startup
logger.info("=" * 50)
logger.info("MODEL LOADER INITIALIZATION")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"ACTIVE_DIR: {ACTIVE_DIR}")
logger.info(f"TensorFlow version: {tf.__version__}")

# Check if active directory exists and list contents
if os.path.exists(ACTIVE_DIR):
    files = os.listdir(ACTIVE_DIR)
    logger.info(f"Files in active directory ({len(files)}): {files}")
    
    # Check for the specific .keras file
    keras_file_path = os.path.join(ACTIVE_DIR, KERAS_FILE)
    if os.path.exists(keras_file_path):
        file_size = os.path.getsize(keras_file_path)
        logger.info(f"Found {KERAS_FILE} - Size: {file_size} bytes")
        logger.info(f"File permissions: Readable={os.access(keras_file_path, os.R_OK)}")
    else:
        logger.warning(f"{KERAS_FILE} NOT FOUND in active directory!")
else:
    logger.error(f"ACTIVE_DIR does not exist: {ACTIVE_DIR}")
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    logger.info(f"Created ACTIVE_DIR: {ACTIVE_DIR}")
logger.info("=" * 50)


def get_active_version() -> str:
    """Get the current active model version."""
    version_file = os.path.join(ACTIVE_DIR, "version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            version = f.read().strip()
            logger.info(f"Active version from file: {version}")
            return version
    logger.info("No version.txt found, using default v1.0")
    return "v1.0"


def _load_from_disk():
    """Load the model from disk with fallback options."""
    global _model, _model_version
    
    logger.info("=" * 50)
    logger.info("LOADING MODEL FROM DISK")
    
    # Ensure directories exist
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    
    # Define all possible model file paths
    keras_paths = [
        os.path.join(ACTIVE_DIR, KERAS_FILE),      # cattle_final.keras
        os.path.join(ACTIVE_DIR, "model.keras"),    # model.keras
    ]
    
    arch_path = os.path.join(ACTIVE_DIR, ARCH_FILE)
    weights_path = os.path.join(ACTIVE_DIR, WEIGHTS_FILE)
    
    # Log all paths we're checking
    logger.info(f"Checking for model files in: {ACTIVE_DIR}")
    logger.info(f"  - {KERAS_FILE}: {os.path.exists(keras_paths[0])}")
    logger.info(f"  - model.keras: {os.path.exists(keras_paths[1])}")
    logger.info(f"  - {ARCH_FILE}: {os.path.exists(arch_path)}")
    logger.info(f"  - {WEIGHTS_FILE}: {os.path.exists(weights_path)}")
    
    # Try to find a valid .keras file
    keras_path = None
    for path in keras_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            logger.info(f"Found .keras file: {path} (size: {file_size} bytes)")
            if file_size > 0:  # Make sure it's not empty
                keras_path = path
                break
            else:
                logger.warning(f"Found but file is empty: {path}")
    
    model = None
    load_error = None
    
    # Method 1: Load from .keras file (preferred)
    if keras_path:
        try:
            logger.info(f"Attempting to load from .keras: {keras_path}")
            
            # Try loading with different options
            try:
                # First attempt: standard load
                model = tf.keras.models.load_model(keras_path)
                logger.info("Successfully loaded .keras file (standard method)")
            except Exception as e1:
                logger.warning(f"Standard load failed: {str(e1)}")
                
                # Second attempt: load without compilation
                logger.info("Attempting to load with compile=False")
                model = tf.keras.models.load_model(keras_path, compile=False)
                logger.info("Successfully loaded .keras file with compile=False")
                
                # Compile the model
                model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )
                logger.info("Model compiled successfully")
                
        except Exception as e:
            load_error = e
            logger.error(f"Failed to load .keras file: {str(e)}")
            logger.info("Will attempt fallback to JSON+weights")
            model = None
    
    # Method 2: Fallback to JSON + weights
    if model is None and os.path.exists(arch_path) and os.path.exists(weights_path):
        try:
            logger.info("Attempting fallback: Loading from JSON + weights")
            
            with open(arch_path, 'r') as f:
                config = json.load(f)
            
            # Fix configuration for compatibility
            def fix_config(obj):
                if isinstance(obj, dict):
                    # Handle InputLayer configuration
                    if obj.get('class_name') == 'InputLayer':
                        cfg = obj.get('config', {})
                        if 'batch_input_shape' in cfg and 'shape' not in cfg:
                            cfg['shape'] = cfg['batch_input_shape'][1:]
                        elif 'shape' not in cfg:
                            cfg['shape'] = (224, 224, 3)
                    
                    # Remove problematic keys
                    for key in ['groups', 'batch_input_shape']:
                        obj.pop(key, None)
                    
                    # Recursively process values
                    for v in obj.values():
                        fix_config(v)
                        
                elif isinstance(obj, list):
                    for item in obj:
                        fix_config(item)
            
            fix_config(config)
            
            # Create model from config
            model = tf.keras.Model.from_config(config)
            logger.info("Model created from config")
            
            # Load weights
            model.load_weights(weights_path)
            logger.info("Weights loaded successfully")
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            logger.info("Model compiled successfully")
            
            # Save as .keras for future loads
            try:
                save_path = os.path.join(ACTIVE_DIR, "model.keras")
                model.save(save_path)
                logger.info(f"Saved model as {save_path} for future loads")
            except Exception as save_err:
                logger.warning(f"Could not save .keras file: {save_err}")
                
        except Exception as e:
            load_error = e
            logger.error(f"Fallback loading failed: {str(e)}")
            model = None
    
    # If no model loaded, raise detailed error
    if model is None:
        error_msg = f"Failed to load model from {ACTIVE_DIR}. "
        
        # List all files for debugging
        if os.path.exists(ACTIVE_DIR):
            files = os.listdir(ACTIVE_DIR)
            error_msg += f"Directory contents: {files}"
        else:
            error_msg += f"Directory does not exist!"
        
        if load_error:
            error_msg += f" Last error: {str(load_error)}"
        
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Store the loaded model
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
        
        # Check for weights file
        weights = os.path.join(folder_path, WEIGHTS_FILE)
        size_kb = os.path.getsize(weights) / 1024 if os.path.exists(weights) else 0
        
        # Get modification time
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
    import datetime
    
    version_path = os.path.join(VERSION_DIR, version)
    if not os.path.isdir(version_path):
        raise ValueError(f"Version '{version}' not found in {VERSION_DIR}")
    
    logger.info(f"Activating version: {version}")
    
    # Create active directory if it doesn't exist
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    
    # Copy model files
    files_copied = []
    for fname in [ARCH_FILE, WEIGHTS_FILE, KERAS_FILE]:
        src = os.path.join(version_path, fname)
        dst = os.path.join(ACTIVE_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            files_copied.append(fname)
            logger.info(f"Copied {fname} to active directory")
    
    # Clean up any existing .keras files to avoid confusion
    for fname in ["model.keras", KERAS_FILE]:
        p = os.path.join(ACTIVE_DIR, fname)
        if os.path.exists(p) and fname not in files_copied:
            os.remove(p)
            logger.info(f"Removed {fname} from active directory")
    
    # Update version file
    with open(os.path.join(ACTIVE_DIR, "version.txt"), "w") as f:
        f.write(version)
    
    # Reload model
    reload_model()
    logger.info(f"Version {version} activated successfully")
    
    return version


# Optional: Pre-load model on import
# Uncomment the line below if you want to load the model immediately when this module is imported
# _load_from_disk()