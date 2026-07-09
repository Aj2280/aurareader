import os
import torch
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# CUDA/MPS Device Patching Shim
# Intercepts hardcoded .cuda() and torch.autocast("cuda", ...) calls inside 
# Hugging Face dynamic module source files to allow execution on CPU or Apple Silicon (MPS).
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch_dtype = torch.bfloat16 if (device == "cuda" or device == "mps") else torch.float32

# Override Tensor.cuda
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device=device, *args, **kwargs)

# Override torch.autocast
if not torch.cuda.is_available():
    original_autocast = torch.autocast
    torch.autococast_is_patched = True
    torch.autocast = lambda device_type, *args, **kwargs: (
        original_autocast("cpu", *args, **kwargs) if device_type == "cuda" else original_autocast(device_type, *args, **kwargs)
    )

MODEL_NAME = "baidu/Unlimited-OCR"
tokenizer = None
model = None

def load_model():
    """Load model and tokenizer dynamically into memory on correct device."""
    global tokenizer, model
    if model is not None:
        return model, tokenizer

    print(f"Loading tokenizer ({MODEL_NAME})...")
    try:
        # Try loading offline fast tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            local_files_only=True
        )
        print("Successfully loaded tokenizer from local cache.")
    except Exception as e_local_fast:
        print(f"Failed loading offline fast tokenizer ({e_local_fast}), trying offline slow tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True, 
                local_files_only=True,
                use_fast=False
            )
            print("Successfully loaded slow tokenizer from local cache.")
        except Exception as e_local_slow:
            print(f"Failed loading offline tokenizer ({e_local_slow}), retrying online...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, 
                    trust_remote_code=True, 
                    local_files_only=False,
                    use_fast=False
                )
            except Exception as e_online:
                raise RuntimeError(f"Failed to load tokenizer: offline error: {e_local_slow}, online error: {e_online}")
    
    print(f"Loading model on device '{device}' with dtype '{torch_dtype}'...")
    try:
        # Try loading offline first
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch_dtype,
            local_files_only=True
        ).eval().to(device)
        print("Successfully loaded model from local cache.")
    except Exception as e_local:
        print(f"Failed loading offline model ({e_local}), retrying online...")
        try:
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch_dtype,
                local_files_only=False
            ).eval().to(device)
        except Exception as e_online:
            raise RuntimeError(f"Failed to load model: offline error: {e_local}, online error: {e_online}")
    
    print("Model loaded successfully.")
    return model, tokenizer
