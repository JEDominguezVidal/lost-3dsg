#!/usr/bin/env python3
"""Script to export EfficientViT-SAM l2 model to ONNX format"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Add the project root and src/perception_module to path
project_root = Path(__file__).resolve().parent
perception_module_path = project_root / "src" / "perception_module"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(perception_module_path))

from efficientvit.export_encoder import EncoderOnnxModel
from efficientvit.models.efficientvit.sam import EfficientViTSam
from efficientvit.sam_model_zoo import create_sam_model


def export_encoder(model: EfficientViTSam, output_path: str, opset: int = 17) -> None:
    """Export the encoder to ONNX"""
    onnx_model = EncoderOnnxModel(model=model)
    
    image_size = [512, 512]
    dummy_input = {"input_image": torch.randn((1, 3, image_size[0], image_size[1]), dtype=torch.float)}
    dynamic_axes = {
        "input_image": {0: "batch_size"},
    }
    
    _ = onnx_model(**dummy_input)
    
    output_names = ["image_embeddings"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting encoder to {output_path}...")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_input.values()),
            output_path,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(dummy_input.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print(f"Encoder export completed: {output_path}")


def export_decoder(model: EfficientViTSam, output_path: str, opset: int = 17) -> None:
    """Export the decoder to ONNX"""
    class DecoderOnnxModel(torch.nn.Module):
        def __init__(self, model: EfficientViTSam):
            super().__init__()
            self.model = model
            self.mask_decoder = self.model.mask_decoder
            self.prompt_encoder = self.model.prompt_encoder
            
        @torch.no_grad()
        def forward(self, image_embeddings, point_coords, point_labels):
            # Convert point coords and labels to the format expected by the prompt encoder
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            
            return low_res_masks, iou_predictions
            
    onnx_model = DecoderOnnxModel(model=model)
    
    # Dummy inputs
    dummy_image_embeddings = torch.randn(1, 256, 64, 64)
    dummy_point_coords = torch.randn(1, 2, 2)
    dummy_point_labels = torch.randn(1, 2)
    
    output_names = ["low_res_masks", "iou_predictions"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting decoder to {output_path}...")
        torch.onnx.export(
            onnx_model,
            (dummy_image_embeddings, dummy_point_coords, dummy_point_labels),
            output_path,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=output_names,
            dynamic_axes={
                "point_coords": {0: "batch_size"},
                "point_labels": {0: "batch_size"},
                "low_res_masks": {0: "batch_size"},
                "iou_predictions": {0: "batch_size"},
            },
        )
    print(f"Decoder export completed: {output_path}")


def main():
    print("Loading EfficientViTSam l2 model...")
    model = create_sam_model("l2", pretrained=True)
    model.eval()
    
    # Output directory for ONNX models
    output_dir = Path(__file__).resolve().parent / "src/perception_module"
    encoder_output = output_dir / "utils" / "l2_encoder.onnx"
    decoder_output = output_dir / "utils" / "l2_decoder.onnx"
    
    # Create the utils directory if it doesn't exist
    encoder_output.parent.mkdir(exist_ok=True)
    decoder_output.parent.mkdir(exist_ok=True)
    
    export_encoder(model, str(encoder_output))
    export_decoder(model, str(decoder_output))
    
    print("\n✅ Export completed successfully!")
    print(f"Encoder: {encoder_output}")
    print(f"Decoder: {decoder_output}")


if __name__ == "__main__":
    main()