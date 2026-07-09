import os
import tempfile
import shutil
import fitz

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert every page of a PDF to a PNG. Returns list of file paths."""
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    try:
        for i, page in enumerate(doc):
            out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out)
            paths.append(out)
    except Exception as e:
        # Clean up partial directory if something goes wrong
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e
    finally:
        doc.close()
    return paths

def clean_directory(dir_path: str):
    """Safely delete a directory and all of its contents."""
    if dir_path and os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
