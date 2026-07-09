try:
    import spaces
except ImportError:
    pass

import os
import sys
import subprocess

# Add current directory to Python path to ensure clean modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Pinned runtime dependencies for Hugging Face Spaces to avoid build-time conflicts
_RUNTIME_PKGS = [
    "torch==2.10.0",
    "torchvision==0.25.0",
    "transformers==4.57.1",
]

if "SPACE_ID" in os.environ:
    print("Installing pinned runtime dependencies on Hugging Face Spaces...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"] + _RUNTIME_PKGS,
            check=True,
        )
        print("Runtime dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing runtime dependencies: {e}", file=sys.stderr)

from api.routes import app
import gradio as gr

custom_css = """
footer {display: none !important;}
#title-header {
    text-align: center;
    background: linear-gradient(135deg, #ff6b35 0%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
"""

if __name__ == "__main__":
    app.launch(
        show_error=True,
        theme=gr.themes.Default(primary_hue="orange", neutral_hue="slate"),
        css=custom_css
    )
