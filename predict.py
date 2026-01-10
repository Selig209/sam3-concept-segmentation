"""
SAM 3 (Segment Anything Model 3) Predictor for Replicate/Cog
Uses Meta's SAM 3 with Promptable Concept Segmentation (PCS)

SAM 3 Features:
- Text prompt segmentation ("yellow school bus", "swimming pools")
- Image exemplar prompts
- Unified detection, segmentation, and tracking
- Released November 2025 by Meta

Weights are downloaded from Hugging Face: facebook/sam3
"""

from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
import os
import json
from shapely.geometry import shape, mapping
from rasterio import features
import cv2

class Predictor(BasePredictor):
    def setup(self):
        """Load SAM 3 model weights from Hugging Face"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Download SAM 3 weights from Hugging Face
        from huggingface_hub import hf_hub_download
        
        # SAM 3 checkpoint from Meta's official release
        model_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            cache_dir="/tmp/sam3_cache"
        )
        print(f"SAM 3 weights downloaded to: {model_path}")
        
        # Load SAM 3 model
        # SAM 3 uses a DETR-based architecture with concept prompting
        try:
            from sam3 import SAM3, SAM3Config
            
            config = SAM3Config()
            self.model = SAM3(config)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(self.device)
            self.model.eval()
            print("SAM 3 model loaded successfully")
            
        except ImportError:
            # Fallback: Try using the Ultralytics implementation
            print("Using Ultralytics SAM implementation...")
            from ultralytics import SAM
            self.model = SAM(model_path)
            self.use_ultralytics = True
            
    def predict(
        self,
        image: Path = Input(description="Input satellite/aerial image to segment"),
        prompt: str = Input(
            description="Concept prompt - describe objects to extract (e.g., 'swimming pools', 'solar panels', 'yellow school bus')", 
            default=""
        ),
        threshold: float = Input(description="Confidence threshold for detections", default=0.5),
        box_threshold: float = Input(description="Box detection threshold", default=0.3),
        text_threshold: float = Input(description="Text matching threshold", default=0.25)
    ) -> dict:
        """
        Run SAM 3 with Promptable Concept Segmentation (PCS)
        
        SAM 3 can detect and segment ALL instances of a visual concept
        specified by short noun phrases, image exemplars, or both.
        """
        
        # 1. Load image
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)
        h, w, _ = img_np.shape
        print(f"Processing image: {w}x{h} with prompt: '{prompt}'")
        
        # 2. Run SAM 3 inference
        if hasattr(self, 'use_ultralytics') and self.use_ultralytics:
            # Ultralytics SAM path
            results = self.model(
                img_np,
                prompts=[prompt] if prompt else None,
                conf=threshold
            )
            masks = results[0].masks.data.cpu().numpy() if results[0].masks else np.array([])
        else:
            # Native SAM 3 path
            with torch.no_grad():
                # Prepare input
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)
                
                # Run concept segmentation
                outputs = self.model.predict_concepts(
                    images=img_tensor,
                    text_prompts=[prompt] if prompt else None,
                    confidence_threshold=threshold,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                
                masks = outputs["masks"].cpu().numpy()
        
        # 3. Post-processing cleanup
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
            
            mask = mask.astype(np.uint8)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.medianBlur(mask, 3)
            
            combined_mask = np.maximum(combined_mask, mask * (i + 1))
        
        # 4. Create colored visualization mask
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Cyan color for SAM 3 detections
        mask_rgba[combined_mask > 0] = [0, 200, 200, 180]
        
        # 5. Vectorize to GeoJSON
        geojson_features = []
        
        if np.any(combined_mask):
            shapes_generator = features.shapes(
                combined_mask,
                mask=combined_mask > 0,
                transform=features.transform.Affine(1, 0, 0, 0, -1, h)
            )
            
            for geom, value in shapes_generator:
                if value > 0:
                    geojson_features.append({
                        "type": "Feature",
                        "properties": {
                            "class": prompt if prompt else "detected_object",
                            "instance_id": int(value),
                            "confidence": float(threshold)
                        },
                        "geometry": geom
                    })
        
        feature_count = len(geojson_features)
        print(f"Extracted {feature_count} features for concept: '{prompt}'")
        
        # 6. Save outputs
        out_mask_path = Path("/tmp/sam3_mask.png")
        Image.fromarray(mask_rgba).save(str(out_mask_path))
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_features
        }
        out_geojson_path = Path("/tmp/sam3_output.geojson")
        with open(str(out_geojson_path), "w") as f:
            json.dump(geojson_data, f)
        
        # Return results
        results = []
        if feature_count > 0:
            results.append({
                "feature_class": prompt if prompt else "detected_object",
                "feature_count": feature_count,
                "geojson": geojson_data,
                "class_mask_path": str(out_mask_path)
            })
        
        return {
            "mask": out_mask_path,
            "geojson": out_geojson_path,
            "results": results
        }
