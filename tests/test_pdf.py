import os
import sys

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdf import pdf_to_images, clean_directory

def test_clean_directory_nonexistent():
    # Calling clean_directory on non-existent path should not raise errors
    clean_directory("non_existent_directory_path_123")
    assert True
