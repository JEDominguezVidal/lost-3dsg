#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class OWLv2():
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_id, local_files_only=False)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=False)
        self.model.to(self.device)
        self.model.eval()
        self.classes = None

    
    def set_classes(self, classes):
        # OWL-ViT works with natural language queries
        self.classes = [cls.lower().strip() for cls in classes]
    
    def predict(self, image, box_threshold=0.1, text_threshold=0.1):
        if self.classes is None:
            raise ValueError("Call set_classes before predict().")
        
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        else:
            image_pil = image
        
        # Prepare text queries
        text_queries = [[f"a photo of a {cls}" for cls in self.classes]]
        
        # Process inputs
        inputs = self.processor(
            text=text_queries, 
            images=image_pil, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=box_threshold,
            target_sizes=target_sizes
        )[0]
        
        bboxes, classes, confidences = [], [], []
        
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            if score >= box_threshold:
                xmin, ymin, xmax, ymax = box.cpu().tolist()
                bboxes.append([xmin, ymin, xmax, ymax])
                classes.append(self.classes[label])
                confidences.append(float(score))
        
        return bboxes, classes, confidences
    
    def get_image_with_bboxes(self, image, conf=0.1):
        bboxes, classes, confidences = self.predict(image, box_threshold=conf)
        
        for i in range(len(bboxes)):
            if confidences[i] >= conf:
                x1, y1, x2, y2 = map(int, bboxes[i])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image, 
                    f"{classes[i]} {confidences[i]:.2f}", 
                    (x1, max(0, y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
        
        return image


class DINO():
    def __init__(self, model_id="IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=False).to(self.device)
        self.classes = None

    def set_classes(self, classes):
        # GroundingDINO expects "." at the end of each query
        self.classes = [cls.lower().strip() + "." for cls in classes]

    def predict(self, image, box_threshold=0.4, text_threshold=0.3):
        if self.classes is None:
            raise ValueError("Call set_classes before predict().")

        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = Image.fromarray(image_rgb)
        else:
            image_rgb = image

        text_queries = " ".join(self.classes)

        inputs = self.processor(images=image_rgb, text=text_queries, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_rgb.size[::-1]] 
        )[0]

        bboxes, classes, confidences = [], [], []
        for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            bboxes.append([xmin, ymin, xmax, ymax])
            classes.append(label_id)  
            confidences.append(float(score))
        return bboxes, classes, confidences
    
    def get_image_with_bboxes(self, image, conf=0.4): 
        bboxes, classes, confidences = self.predict(image, box_threshold=conf) 
        for i in range(len(bboxes)):
             if confidences[i] >= conf: 
                x1, y1, x2, y2 = map(int, bboxes[i]) 
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.putText(image, f"{classes[i]} {confidences[i]:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        return image


class VitSam():
    """
    EfficientViT SAM wrapper using native PyTorch inference.
    
    Note: ONNX export had issues producing NaN outputs, so this class
    uses PyTorch directly via EfficientViTSamPredictor for reliable inference.
    """

    def __init__(self, encoder_model=None, decoder_model=None):
        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"VitSam device: {self.device} (PyTorch mode)")
        
        # Load the full EfficientViT SAM model using PyTorch
        import sys
        import os
        # Add path for efficientvit imports
        module_path = os.path.dirname(os.path.abspath(__file__))
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
        
        from efficientvit.sam_model_zoo import create_sam_model
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        
        # Compute absolute path to checkpoint (project root / assets / checkpoints / sam / l2.pt)
        # module_path = src/perception_module/, go up 2 levels to reach project root (lost-3dsg/)
        project_root = os.path.dirname(os.path.dirname(module_path))
        weight_path = os.path.join(project_root, "assets", "checkpoints", "sam", "l2.pt")
        
        print(f"Loading EfficientViT SAM l2 model from {weight_path}...")
        sam_model = create_sam_model("l2", pretrained=True, weight_url=weight_path).eval()
        sam_model = sam_model.to(self.device)
        
        # Create the predictor wrapper
        self.predictor = EfficientViTSamPredictor(sam_model)
        print("Model loaded successfully!")
        
        # Image size for l2 model
        self.img_size = 512
        self.target_size = 512

    def __call__(self, img, bboxes):
        """
        Generate masks for given bounding boxes.
        
        Args:
            img: BGR image (numpy array)
            bboxes: List of [x1, y1, x2, y2] bounding boxes
            
        Returns:
            masks: Tensor of shape [N, 1, H, W] with boolean masks
            boxes: Input boxes as numpy array
        """
        # Convert to RGB for the predictor
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        
        # Set image (computes embeddings)
        self.predictor.set_image(raw_img)
        
        boxes = np.array(bboxes, dtype=np.float32)
        
        # Generate masks for each box
        all_masks = []
        for box in boxes:
            # Predict mask for this box
            masks_np, iou_pred, low_res_masks = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            
            # Convert to torch tensor with shape [1, H, W] -> [1, 1, H, W]
            mask_tensor = torch.from_numpy(masks_np).to(self.device)
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dim
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dim [1, 1, H, W]
            all_masks.append(mask_tensor)
        
        # Reset predictor for next call
        self.predictor.reset_image()
        
        # Stack all masks
        if all_masks:
            masks = torch.cat(all_masks, dim=0)  # [N, 1, H, W]
        else:
            masks = torch.zeros((0, 1, origin_image_size[0], origin_image_size[1]), 
                               dtype=torch.bool, device=self.device)
        
        return masks, boxes