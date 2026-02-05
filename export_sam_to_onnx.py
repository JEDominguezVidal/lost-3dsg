#!/usr/bin/env python3
"""Script to export EfficientViT-SAM l2 model to ONNX format.

This script exports the EfficientViT-SAM encoder and decoder to ONNX format
with an ONNX-compatible decoder wrapper that avoids boolean indexing issues.
"""

import os
import sys
import torch
import torch.nn as nn
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

# Use opset 18 - PyTorch 2.10+ cannot reliably downgrade to 17
# due to missing adapters for Pad and ScatterElements operators
ONNX_OPSET_VERSION = 18


class OnnxCompatiblePromptEncoder(nn.Module):
    """
    ONNX-compatible wrapper for SAM's PromptEncoder.
    
    The original PromptEncoder uses boolean indexing like:
        point_embedding[labels == -1] = 0.0
    
    This translates to ScatterElements/Where operators in ONNX that have
    broadcasting issues. This wrapper reimplements the logic using
    torch.where() with explicit broadcasting.
    """
    
    def __init__(self, prompt_encoder):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        
    def _embed_points_onnx_safe(self, points, labels):
        """
        ONNX-safe implementation of point embedding.
        Avoids boolean indexing by using torch.where() with masks.
        """
        # Shift to center of pixel
        points = points + 0.5
        
        # Get positional encoding for the points
        point_embedding = self.prompt_encoder.pe_layer.forward_with_coords(
            points, self.prompt_encoder.input_image_size
        )
        
        # Get embedding weights
        not_a_point_embed = self.prompt_encoder.not_a_point_embed.weight  # [1, embed_dim]
        point_embed_0 = self.prompt_encoder.point_embeddings[0].weight  # neg point
        point_embed_1 = self.prompt_encoder.point_embeddings[1].weight  # pos point
        point_embed_2 = self.prompt_encoder.point_embeddings[2].weight  # box corner 1
        point_embed_3 = self.prompt_encoder.point_embeddings[3].weight  # box corner 2
        
        # Create masks for each label type - shape: [B, N, 1]
        labels_expanded = labels.unsqueeze(-1).float()
        
        # For label -1 (not a point): zero out and add not_a_point_embed
        mask_neg1 = (labels_expanded == -1).float()
        # For label 0 (negative point)
        mask_0 = (labels_expanded == 0).float()
        # For label 1 (positive point)  
        mask_1 = (labels_expanded == 1).float()
        # For label 2 (box corner top-left)
        mask_2 = (labels_expanded == 2).float()
        # For label 3 (box corner bottom-right)
        mask_3 = (labels_expanded == 3).float()
        
        # Apply embeddings using masks (no boolean indexing!)
        # Zero out where label == -1, keep positional encoding otherwise
        point_embedding = point_embedding * (1.0 - mask_neg1)
        
        # Add the appropriate embedding based on label
        point_embedding = point_embedding + mask_neg1 * not_a_point_embed
        point_embedding = point_embedding + mask_0 * point_embed_0
        point_embedding = point_embedding + mask_1 * point_embed_1
        point_embedding = point_embedding + mask_2 * point_embed_2
        point_embedding = point_embedding + mask_3 * point_embed_3
        
        return point_embedding
    
    def forward(self, point_coords, point_labels):
        """
        Forward pass using ONNX-safe point embedding.
        
        Args:
            point_coords: [B, N, 2] - point coordinates
            point_labels: [B, N] - point labels (0=neg, 1=pos, 2=box_tl, 3=box_br, -1=pad)
        
        Returns:
            sparse_embeddings: [B, N, embed_dim]
            dense_embeddings: [B, embed_dim, H, W]
        """
        bs = point_coords.shape[0]
        
        # Get point embeddings using our ONNX-safe method
        sparse_embeddings = self._embed_points_onnx_safe(point_coords, point_labels)
        
        # Dense embeddings (no mask input, so use no_mask_embed)
        dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, 
            self.prompt_encoder.image_embedding_size[0], 
            self.prompt_encoder.image_embedding_size[1]
        )
        
        return sparse_embeddings, dense_embeddings
    
    def get_dense_pe(self):
        return self.prompt_encoder.get_dense_pe()


class DecoderOnnxModel(nn.Module):
    """ONNX-compatible SAM decoder model."""
    
    def __init__(self, model: EfficientViTSam):
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.onnx_prompt_encoder = OnnxCompatiblePromptEncoder(model.prompt_encoder)
        
    @torch.no_grad()
    def forward(self, image_embeddings, point_coords, point_labels):
        """
        Forward pass for ONNX export.
        
        Args:
            image_embeddings: [B, 256, 64, 64] - encoded image features
            point_coords: [B, N, 2] - point/box coordinates
            point_labels: [B, N] - point labels
        
        Returns:
            low_res_masks: [B, 1, 256, 256] - low resolution masks
            iou_predictions: [B, 1] - IoU predictions
        """
        # Convert labels to float for the ONNX-safe encoder
        point_labels_float = point_labels.float()
        
        # Get embeddings using ONNX-safe encoder
        sparse_embeddings, dense_embeddings = self.onnx_prompt_encoder(
            point_coords, point_labels_float
        )
        
        # Run mask decoder
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.onnx_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return low_res_masks, iou_predictions


def export_encoder(model: EfficientViTSam, output_path: str, opset: int = ONNX_OPSET_VERSION) -> None:
    """Export the encoder to ONNX."""
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
        print(f"Exporting encoder to {output_path} (opset {opset})...")
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


def export_decoder(model: EfficientViTSam, output_path: str, opset: int = ONNX_OPSET_VERSION) -> None:
    """Export the decoder to ONNX using the ONNX-compatible wrapper."""
    onnx_model = DecoderOnnxModel(model=model)
    onnx_model.eval()
    
    # Dummy inputs - use box labels (2, 3) to test the most common use case
    dummy_image_embeddings = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    dummy_point_coords = torch.randn(1, 2, 2, dtype=torch.float32)
    dummy_point_labels = torch.tensor([[2, 3]], dtype=torch.int64)  # Box corners
    
    output_names = ["low_res_masks", "iou_predictions"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting decoder to {output_path} (opset {opset})...")
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
                "image_embeddings": {0: "batch_size"},
                "point_coords": {0: "batch_size", 1: "num_points"},
                "point_labels": {0: "batch_size", 1: "num_points"},
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