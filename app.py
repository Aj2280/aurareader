import os
import sys

# Add current directory to Python path to ensure clean modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
