#!/usr/bin/env python3
"""
Florence-2 Node for LOST-3DSG

This node runs the Florence-2 model in a separate process (and virtual environment)
to avoid dependency conflicts with the main perception pipeline.

Services:
- /florence2/detect (DetectObjects)
- /florence2/describe (GetImageDescription)
"""

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Import custom service/msg definitions
# Note: These must be compiled first!
from lost3dsg.msg import ObjectDetection2D
from lost3dsg.srv import DetectObjects, GetImageDescription


class Florence2Node(Node):
    MODEL_ID = "microsoft/Florence-2-large"

    def __init__(self):
        super().__init__('florence2_node')
        
        self.declare_parameter('device', 'cuda')
        self.device = self.get_parameter('device').get_parameter_value().string_value
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            
        self.get_logger().info(f"Loading Florence-2-large on {self.device}...")
        
        # Load Model & Processor
        # In 'florence2' venv (transformers 4.x), trust_remote_code=True works out of the box
        # We use explicit float16 for CUDA to match the model weights
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True
            )
            self.model.eval()
            self.get_logger().info("Florence-2 loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise e

        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Create Services
        self.detect_service = self.create_service(
            DetectObjects, 
            '/florence2/detect', 
            self.detect_callback
        )
        
        self.describe_service = self.create_service(
            GetImageDescription, 
            '/florence2/describe', 
            self.describe_callback
        )
        
        self.get_logger().info("Services /florence2/detect and /florence2/describe are ready.")

    def _msg_to_pil(self, msg):
        """Convert ROS Image message to PIL Image."""
        # Force RGB8 encoding
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return None
            
        return Image.fromarray(cv_image)

    def _run_inference(self, image, task_prompt, max_new_tokens=1024):
        """Run Florence-2 inference."""
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.model.dtype)

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

        parsed_result = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_result

    def detect_callback(self, request, response):
        """Handle object detection requests."""
        try:
            image = self._msg_to_pil(request.image)
            if image is None:
                return response
                
            task_prompt = "<OD>"
            
            # Run inference
            result = self._run_inference(image, task_prompt)
            
            response.detections = []
            if "<OD>" in result:
                od_result = result["<OD>"]
                bboxes = od_result.get("bboxes", [])
                labels = od_result.get("labels", [])
                
                for bbox, label in zip(bboxes, labels):
                    det = ObjectDetection2D()
                    det.label = label
                    det.score = 1.0
                    # Ensure bbox is integer list [x1, y1, x2, y2]
                    det.bbox = [int(c) for c in bbox]
                    response.detections.append(det)
            
            self.get_logger().info(f"Detected {len(response.detections)} objects.")
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error in detect_callback: {e}")
            return response

    def describe_callback(self, request, response):
        """Handle image description requests."""
        try:
            image = self._msg_to_pil(request.image)
            if image is None:
                return response
                
            prompt = request.prompt if request.prompt else "<CAPTION>"
            
            # Run inference
            result = self._run_inference(image, prompt)
            
            if prompt in result:
                response.description = result[prompt]
            else:
                response.description = ""
                
            # self.get_logger().info(f"Generated description: {response.description[:50]}...")
            return response

        except Exception as e:
            self.get_logger().error(f"Error in describe_callback: {e}")
            return response


def main(args=None):
    rclpy.init(args=args)
    node = Florence2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
