import os
from PIL import Image
import fitz

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_PDF_SIZE = 30 * 1024 * 1024    # 30 MB
MAX_PDF_PAGES = 30

def validate_image_file(file_path: str):
    """Validate that the file is a valid image within size constraints."""
    if not os.path.exists(file_path):
        raise ValueError("File does not exist.")
        
    size = os.path.getsize(file_path)
    if size > MAX_IMAGE_SIZE:
        raise ValueError(f"Image file size ({size / 1024 / 1024:.1f} MB) exceeds maximum limit of {MAX_IMAGE_SIZE / 1024 / 1024:.1f} MB.")
        
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Invalid image format or corrupted image file: {str(e)}")

def validate_pdf_file(file_path: str):
    """Validate that the file is a valid PDF within size and page count constraints."""
    if not os.path.exists(file_path):
        raise ValueError("File does not exist.")
        
    size = os.path.getsize(file_path)
    if size > MAX_PDF_SIZE:
        raise ValueError(f"PDF file size ({size / 1024 / 1024:.1f} MB) exceeds maximum limit of {MAX_PDF_SIZE / 1024 / 1024:.1f} MB.")
        
    try:
        doc = fitz.open(file_path)
        pages = len(doc)
        doc.close()
    except Exception as e:
        raise ValueError(f"Invalid PDF or corrupted PDF file: {str(e)}")
        
    if pages > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {pages} pages, which exceeds the limit of {MAX_PDF_PAGES} pages.")
    
    return pages
