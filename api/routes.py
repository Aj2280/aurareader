import os
import queue
import tempfile
import threading
from typing import Iterator
import gradio as gr

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(duration=None):
            def decorator(func):
                return func
            return decorator

# Core imports
from core.model import load_model, device
from core.streaming import register_thread_queue, unregister_thread_queue, patch_model_generation
from core.pdf import pdf_to_images, clean_directory
from core.validation import validate_image_file, validate_pdf_file

def _collect_output(out_dir: str) -> str:
    """Read all text/markdown files written by model.infer()."""
    result = ""
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith((".txt", ".md")):
            with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as f:
                result += f.read() + "\n"
    if not result:
        for fname in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        result += f.read() + "\n"
                except Exception:
                    pass
    return result.strip()

@spaces.GPU(duration=60)
def run_ocr_stream(file_path: str, mode: str, prompt: str) -> Iterator[str]:
    """Inference generator that streams OCR text tokens."""
    model, tokenizer = load_model()
    
    if not getattr(model, "generate_is_patched", False):
        patch_model_generation(model, tokenizer)
        model.generate_is_patched = True

    out_dir = tempfile.mkdtemp(prefix="ocr_out_")
    base_size, image_size, crop_mode, ngram_window = (1024, 640, True, 128) if mode == "long" else (1024, 1024, False, 128)

    _infer_kwargs = dict(
        prompt=f"<image>{prompt}",
        image_file=file_path,
        output_path=out_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=8192,
        no_repeat_ngram_size=35,
        ngram_window=ngram_window,
        save_results=True,
    )

    q = queue.Queue()
    errors = []

    def _infer_thread():
        try:
            model.infer(tokenizer, **_infer_kwargs)
        except Exception as e:
            errors.append(str(e))

    thread = threading.Thread(target=_infer_thread, daemon=True)
    register_thread_queue(thread, q)

    accumulated = ""
    try:
        thread.start()
        while thread.is_alive() or not q.empty():
            try:
                chunk = q.get(timeout=0.02)
                accumulated += chunk
                yield accumulated
            except queue.Empty:
                continue
    finally:
        unregister_thread_queue(thread)
        thread.join()

    full_text = _collect_output(out_dir)
    clean_directory(out_dir)

    if full_text:
        yield full_text
    elif accumulated:
        yield accumulated
    else:
        if errors:
            raise RuntimeError(f"Inference failed: {', '.join(errors)}")
        yield ""

def process_document(file, mode, preset, custom_prompt):
    """Processes uploaded file (Image or PDF) and yields streamed text."""
    if not file:
        yield "Please upload a document to begin."
        return

    file_path = file.name
    is_pdf = file_path.lower().endswith('.pdf')
    
    # Select prompt based on preset
    prompt_map = {
        "General OCR": "document parsing.",
        "Table Extraction": "extract all tables to markdown formatting.",
        "Math & Formulas": "extract all mathematical formulas.",
        "Layout & Bounding Boxes": "parse layout with bounding boxes.",
    }
    prompt = custom_prompt if preset == "Custom Prompt" else prompt_map.get(preset, "document parsing.")

    try:
        if is_pdf:
            # Validate PDF
            validate_pdf_file(file_path)
            yield "Extracting pages from PDF..."
            pages = pdf_to_images(file_path, dpi=200)
            
            full_results = []
            for i, page_path in enumerate(pages):
                yield f"--- Processing Page {i+1}/{len(pages)} ---\n"
                page_text = ""
                for stream_text in run_ocr_stream(page_path, mode, prompt):
                    # Show progress to user
                    yield "\n".join(full_results) + f"\n\n--- Page {i+1}/{len(pages)} ---\n" + stream_text
                    page_text = stream_text
                
                full_results.append(f"--- PAGE {i+1} ---\n{page_text}")
                clean_directory(os.path.dirname(page_path)) # clean per-page temp file
                
            yield "\n\n".join(full_results)
        else:
            # Validate Image
            validate_image_file(file_path)
            for stream_text in run_ocr_stream(file_path, mode, prompt):
                yield stream_text
    except Exception as e:
        yield f"Error during processing: {str(e)}"

def update_preview(file):
    """Generate a preview path for an image or extract the first page of a PDF."""
    if not file:
        return gr.update(value=None, visible=False)
        
    file_path = file.name
    if file_path.lower().endswith('.pdf'):
        try:
            import fitz
            doc = fitz.open(file_path)
            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                temp_dir = tempfile.gettempdir()
                out_path = os.path.join(temp_dir, "pdf_preview_first_page.png")
                pix.save(out_path)
                doc.close()
                return gr.update(value=out_path, visible=True)
            doc.close()
        except Exception:
            pass
        return gr.update(value=None, visible=False)
    else:
        return gr.update(value=file_path, visible=True)

def generate_export(text, export_format):
    """Generate a temporary file of the OCR output in the requested format."""
    if not text:
        return gr.update(value=None, visible=False)
        
    ext = ".txt"
    if "Markdown" in export_format:
        ext = ".md"
    elif "HTML" in export_format:
        ext = ".html"
        
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    # Wrap in basic HTML structure if HTML is selected
    if ext == ".html":
        html_content = f"<html><head><meta charset='utf-8'><title>AuraReader Pro OCR Export</title></head><body><pre style='white-space: pre-wrap;'>{text}</pre></body></html>"
        tmp_file.write(html_content.encode("utf-8"))
    else:
        tmp_file.write(text.encode("utf-8"))
    tmp_file.close()
    return gr.update(value=tmp_file.name, visible=True)

def update_stats(text):
    """Compute character and word count statistics for the output."""
    if not text:
        return "ℹ️ **Statistics**: 0 characters | 0 words"
    chars = len(text)
    words = len(text.split())
    return f"ℹ️ **Statistics**: {chars:,} characters | {words:,} words"

# Custom CSS for AuraReader Pro styling
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

# Pre-load model at startup to avoid network requests inside spaces.GPU sandbox
print("Pre-loading model and tokenizer at startup...")
try:
    load_model()
except Exception as e:
    print(f"Error pre-loading model at startup: {e}")

# Build standard Gradio blocks UI
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=12):
            gr.Markdown("# ⚡ AuraReader Pro", elem_id="title-header")
            gr.Markdown("### VLM Document Parser — Personalized for Abhi")

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload Document (Image or PDF)", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".avif"])
            
            preview_image = gr.Image(label="Document Preview", visible=False, type="filepath")
            input_file.change(fn=update_preview, inputs=input_file, outputs=preview_image)
            
            with gr.Row():
                mode_select = gr.Radio(
                    choices=["long", "base"], 
                    value="long", 
                    label="Inference Mode", 
                    info="Long is optimized for speed (640px), Base for high-resolution (1024px)"
                )
            
            preset_select = gr.Dropdown(
                choices=["General OCR", "Table Extraction", "Math & Formulas", "Layout & Bounding Boxes", "Custom Prompt"],
                value="General OCR",
                label="Prompt Preset"
            )
            
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt", 
                value="document parsing.", 
                visible=False,
                placeholder="Enter custom instructions for the model..."
            )
            
            # Show/hide custom prompt box based on preset selection
            def update_preset_visibility(preset):
                return gr.update(visible=(preset == "Custom Prompt"))
            
            preset_select.change(
                fn=update_preset_visibility,
                inputs=preset_select,
                outputs=custom_prompt_input
            )

            with gr.Row():
                start_btn = gr.Button("⚡ Start OCR", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Page")
            
            with gr.Accordion("ℹ️ How to Use & Tips", open=False):
                gr.Markdown("""
                1. **Upload** a PDF or Image.
                2. Choose the **Inference Mode**:
                   * **Long**: Fastest speed (default).
                   * **Base**: Highest precision.
                3. Choose a **Prompt Preset** depending on the document type (tables, math formulas, layout analysis).
                4. Click **Start OCR** to begin streaming.
                5. Once complete, select your **Export Format** in the right column and click **Generate Export Link** to download.
                """)
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="OCR Output", 
                placeholder="OCR output will stream here...", 
                lines=20
            )
            
            stats_md = gr.Markdown("ℹ️ **Statistics**: 0 characters | 0 words")
            output_text.change(fn=update_stats, inputs=output_text, outputs=stats_md)

            with gr.Row():
                format_select = gr.Dropdown(
                    choices=["Plain Text (.txt)", "Markdown (.md)", "HTML (.html)"], 
                    value="Markdown (.md)", 
                    label="Export Format"
                )
                export_btn = gr.Button("💾 Generate Export Link")
            
            download_file = gr.File(label="Download File Link", visible=False)

    # Bind actions
    start_btn.click(
        fn=process_document,
        inputs=[input_file, mode_select, preset_select, custom_prompt_input],
        outputs=output_text
    )

    # Clear input/output fields
    def clear_fields():
        return None, "long", "General OCR", "document parsing.", "", "ℹ️ **Statistics**: 0 characters | 0 words", gr.update(value=None, visible=False), gr.update(value=None, visible=False)

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[input_file, mode_select, preset_select, custom_prompt_input, output_text, stats_md, download_file, preview_image]
    )

    # Export formatting download
    export_btn.click(
        fn=generate_export,
        inputs=[output_text, format_select],
        outputs=download_file
    )
