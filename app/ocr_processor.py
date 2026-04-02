# app/ocr_processor.py

import os
import requests
import uuid
from typing import Optional, List, Tuple
import easyocr
import numpy as np
from PIL import Image
import io

class OCRProcessor:
    """
    Handles OCR processing for medical images using EasyOCR.
    Supports processing images from URLs (Twilio MMS) or local files.
    """
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False, download_dir: str = "./downloaded_images"):
        """
        Initialize EasyOCR reader.
        
        Args:
            languages: List of language codes to support (default: English only)
            gpu: Whether to use GPU acceleration (requires CUDA)
            download_dir: Directory to save downloaded images
        """
        print(f"Initializing EasyOCR with languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        print("✅ EasyOCR initialized successfully")
    
    def download_image(self, image_url: str, auth: Optional[Tuple[str, str]] = None) -> Optional[str]:
        """
        Download image from URL (typically Twilio MMS).
        
        Args:
            image_url: URL of the image
            auth: Optional tuple of (username, password) for authentication
        
        Returns:
            Local file path of downloaded image, or None if failed
        """
        try:
            print(f"Downloading image from: {image_url}")
            response = requests.get(image_url, auth=auth, timeout=30)
            response.raise_for_status()
            
            # Generate unique filename
            file_extension = self._get_extension_from_url(image_url)
            filename = f"{uuid.uuid4()}{file_extension}"
            filepath = os.path.join(self.download_dir, filename)
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Image downloaded: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return None
    
    def _get_extension_from_url(self, url: str) -> str:
        """Extract file extension from URL or default to .jpg"""
        common_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        url_lower = url.lower()
        for ext in common_extensions:
            if ext in url_lower:
                return ext
        return '.jpg'
    
    def extract_text_from_file(self, image_path: str, detail: int = 1) -> dict:
        """
        Extract text from local image file.
        
        Args:
            image_path: Path to the image file
            detail: OCR detail level (0=simple text, 1=detailed with confidence)
        
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            print(f"Processing image with EasyOCR: {image_path}")
            
            # Read image and perform OCR
            results = self.reader.readtext(image_path, detail=detail)
            
            # Extract all text
            if detail == 0:
                # Simple mode: just text strings
                extracted_text = " ".join(results)
            else:
                # Detailed mode: list of (bbox, text, confidence)
                extracted_text = " ".join([text for (bbox, text, conf) in results])
                confidences = [conf for (bbox, text, conf) in results]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"✅ Extracted {len(results)} text elements")
            
            return {
                "success": True,
                "text": extracted_text.strip(),
                "raw_results": results if detail == 1 else None,
                "confidence": avg_confidence if detail == 1 else None,
                "image_path": image_path
            }
            
        except Exception as e:
            print(f"❌ Error during OCR: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "image_path": image_path
            }
    
    def extract_text_from_url(self, image_url: str, auth: Optional[Tuple[str, str]] = None, detail: int = 1) -> dict:
        """
        Download image from URL and extract text.
        
        Args:
            image_url: URL of the image
            auth: Optional authentication tuple
            detail: OCR detail level
        
        Returns:
            Dictionary with extracted text and metadata
        """
        # Download image first
        local_path = self.download_image(image_url, auth=auth)
        
        if not local_path:
            return {
                "success": False,
                "text": "",
                "error": "Failed to download image",
                "image_url": image_url
            }
        
        # Extract text from downloaded image
        result = self.extract_text_from_file(local_path, detail=detail)
        result["image_url"] = image_url
        
        return result
    
    def process_medical_document(self, image_source: str, is_url: bool = True, auth: Optional[Tuple[str, str]] = None) -> dict:
        """
        Process a medical document image with enhanced extraction.
        
        Args:
            image_source: URL or file path of the image
            is_url: Whether the source is a URL or local file
            auth: Authentication for URL downloads
        
        Returns:
            Structured data extracted from the medical document
        """
        # Extract text
        if is_url:
            result = self.extract_text_from_url(image_source, auth=auth, detail=1)
        else:
            result = self.extract_text_from_file(image_source, detail=1)
        
        if not result["success"]:
            return result
        
        # Add medical document classification
        text = result["text"].lower()
        doc_type = self._classify_medical_document(text)
        result["document_type"] = doc_type
        
        return result
    
    def _classify_medical_document(self, text: str) -> str:
        """
        Classify the type of medical document based on extracted text.
        
        Returns:
            Document type (prescription, lab_report, insurance_card, etc.)
        """
        keywords = {
            "prescription": ["prescription", "rx", "medication", "dosage", "pharmacy"],
            "lab_report": ["lab", "test", "result", "blood", "glucose", "cholesterol"],
            "insurance_card": ["insurance", "member id", "group", "policy", "coverage"],
            "medical_bill": ["bill", "invoice", "charges", "payment", "amount due"],
            "discharge_summary": ["discharge", "admission", "diagnosis", "treatment plan"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for doc_type, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return "unknown"
    
    def batch_process(self, image_sources: List[str], is_url: bool = True, auth: Optional[Tuple[str, str]] = None) -> List[dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_sources: List of image URLs or file paths
            is_url: Whether sources are URLs
            auth: Authentication for URL downloads
        
        Returns:
            List of extraction results
        """
        results = []
        for source in image_sources:
            result = self.process_medical_document(source, is_url=is_url, auth=auth)
            results.append(result)
        
        return results
    
    def cleanup_old_images(self, days: int = 7):
        """
        Remove downloaded images older than specified days.
        
        Args:
            days: Age threshold in days
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (days * 86400)  # Convert days to seconds
        
        deleted_count = 0
        for filename in os.listdir(self.download_dir):
            filepath = os.path.join(self.download_dir, filename)
            if os.path.isfile(filepath):
                file_time = os.path.getmtime(filepath)
                if file_time < cutoff_time:
                    os.remove(filepath)
                    deleted_count += 1
        
        print(f"🗑️ Cleaned up {deleted_count} old image(s)")
