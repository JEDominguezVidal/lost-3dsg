#!/usr/bin/env python3
"""
Florence-2 Detector Wrapper for LOST-3DSG

This module provides a wrapper around the Florence-2-large model for:
- Object detection with bounding boxes
- Region captioning/description

Uses native transformers 5.0 support (Florence2ForConditionalGeneration).
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import re


class Florence2Detector:
    """
    Wrapper for Microsoft's Florence-2-large model.
    
    Provides unified interface for object detection and captioning tasks.
    Uses native transformers support (no trust_remote_code needed).
    """
    
    MODEL_ID = "microsoft/Florence-2-large"
    
    def __init__(self, device="cuda"):
        """
        Initialize Florence-2 model.
        
        Args:
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Florence-2-large model on {self.device}...")
        
        # Load model using AutoModelForCausalLM with trust_remote_code=True
        # Note: We will handle the transformers 5.0 compatibility patching manually if needed
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )
        
        # Enable eval mode for inference
        self.model.eval()
        
        print(f"Florence-2-large loaded successfully on {self.device}")
    
    def _numpy_to_pil(self, image):
        """Convert numpy array to PIL Image if needed."""
        if isinstance(image, np.ndarray):
            # Input from ROS perception.py is already RGB (via cv_bridge "rgb8")
            # Do NOT flip channels or copy unless necessary
            return Image.fromarray(image.astype('uint8'))
        return image
    
    def _run_inference(self, image, task_prompt, max_new_tokens=1024):
        """
        Run inference with Florence-2.
        
        Args:
            image: PIL Image or numpy array
            task_prompt: Florence-2 task prompt (e.g., "<OD>", "<CAPTION>")
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Parsed result from Florence-2
        """
        image = self._numpy_to_pil(image)
        
        # Save original size for post-processing scaling
        orig_w, orig_h = image.width, image.height
        
        # Resize to expected size (e.g. 768x768) to ensure square input for Florence-2
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "size"):
            target_size = self.processor.image_processor.size
            if isinstance(target_size, dict) and "height" in target_size and "width" in target_size:
                w, h = target_size["width"], target_size["height"]
                if image.width != w or image.height != h:
                    image = image.resize((w, h))
            elif isinstance(target_size, int):
                if image.width != target_size or image.height != target_size:
                    image = image.resize((target_size, target_size))
        
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Ensure pixel_values match model dtype (float16 if model is half)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False
            )
        
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # Debug log (optional, prints to stdout which ROS captures)
        # print(f"DEBUG: Florence-2 raw output for {task_prompt}: {generated_text}")
        
        # Post-process: Use ORIGINAL size to get correct coordinates for the input image
        parsed_result = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(orig_w, orig_h)
        )
        
        return parsed_result
    
    def detect_objects(self, image):
        """
        Detect objects in image with bounding boxes.
        
        Args:
            image: PIL Image or numpy array (RGB or BGR)
            
        Returns:
            List of detections, each with:
            - 'label': object class name
            - 'bbox': [x1, y1, x2, y2] in pixel coordinates
            - 'score': confidence score (1.0 for Florence-2)
        """
        result = self._run_inference(image, "<OD>")
        
        detections = []
        
        if "<OD>" in result:
            od_result = result["<OD>"]
            bboxes = od_result.get("bboxes", [])
            labels = od_result.get("labels", [])
            
            for bbox, label in zip(bboxes, labels):
                detections.append({
                    "label": label.strip().lower(),
                    "bbox": [int(coord) for coord in bbox],  # [x1, y1, x2, y2]
                    "score": 1.0  # Florence-2 doesn't provide confidence scores
                })
        
        return detections
    
    def describe_image(self, image):
        """
        Generate a detailed caption for the entire image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            String description of the image
        """
        result = self._run_inference(image, "<MORE_DETAILED_CAPTION>")
        
        if "<MORE_DETAILED_CAPTION>" in result:
            return result["<MORE_DETAILED_CAPTION>"]
        
        return ""
    
    def describe_region(self, image, bbox):
        """
        Generate a description for a specific region (crop) of the image.
        
        Args:
            image: PIL Image or numpy array
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            String description of the region
        """
        image = self._numpy_to_pil(image)
        
        # Crop the region
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cropped = image.crop((x1, y1, x2, y2))
        
        # Get detailed caption of the crop
        result = self._run_inference(cropped, "<MORE_DETAILED_CAPTION>")
        
        if "<MORE_DETAILED_CAPTION>" in result:
            return result["<MORE_DETAILED_CAPTION>"]
        
        return ""
    
    def detect_and_describe(self, image):
        """
        Combined detection and description in one call.
        
        Detects all objects and generates descriptions for each.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of detections with descriptions, each with:
            - 'label': object class name
            - 'bbox': [x1, y1, x2, y2] in pixel coordinates
            - 'score': confidence score
            - 'description': text description of the object
        """
        image = self._numpy_to_pil(image)
        
        # First detect objects
        detections = self.detect_objects(image)
        
        # Then describe each detection
        for det in detections:
            det["description"] = self.describe_region(image, det["bbox"])
        
        return detections


# Singleton instance for reuse
_florence_instance = None


def get_florence_detector(device="cuda"):
    """
    Get singleton instance of Florence2Detector.
    
    Args:
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Florence2Detector instance
    """
    global _florence_instance
    if _florence_instance is None:
        _florence_instance = Florence2Detector(device)
    return _florence_instance
