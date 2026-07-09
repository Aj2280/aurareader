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

    # Force local loading if inside Hugging Face Space GPU sandbox to avoid network blocks
    is_gpu_sandbox = "SPACE_ID" in os.environ and torch.cuda.is_available()
    local_files_only = True if is_gpu_sandbox else False

    print(f"Loading tokenizer ({MODEL_NAME}), local_files_only={local_files_only}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            local_files_only=local_files_only
        )
    except Exception as e:
        if local_files_only:
            print(f"Failed loading offline tokenizer, retrying online: {e}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        else:
            raise e
    
    print(f"Loading model on device '{device}' with dtype '{torch_dtype}', local_files_only={local_files_only}...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only
        ).eval().to(device)
    except Exception as e:
        if local_files_only:
            print(f"Failed loading offline model, retrying online: {e}")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch_dtype
            ).eval().to(device)
        else:
            raise e
    
    print("Model loaded successfully.")
    return model, tokenizer
