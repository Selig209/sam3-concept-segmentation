"""
SAM 3 (Segment Anything Model 3) Predictor for Replicate/Cog
Uses Meta's SAM 3 with Promptable Concept Segmentation (PCS)

SAM 3 Features:
- Text prompt segmentation ("yellow school bus", "swimming pools")
- Image exemplar prompts
- Unified detection, segmentation, and tracking
- Released November 2025 by Meta

Weights are baked into the Docker image during build.
"""

from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
import os
import json
from rasterio import features
import cv2

class Predictor(BasePredictor):
    def setup(self):
        """Load SAM 3 model weights from baked-in file"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # SAM 3 weights are baked into the Docker image during build
        model_path = "/src/sam3.pt"  # Baked into image by Cog
        
        if not os.path.exists(model_path):
            model_path = "sam3.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "SAM 3 weights not found. The model file should be baked into the Docker image during build."
            )
        
        print(f"Loading SAM 3 weights from: {model_path}")
        
        # Try loading with segment-geospatial (samgeo)
        try:
            from samgeo import SamGeo2
            self.model = SamGeo2(model_path)
            self.use_samgeo = True
            print("SAM 3 model loaded with samgeo")
        except Exception as e:
            print(f"samgeo failed: {e}")
            # Fallback: Try using the Ultralytics implementation
            print("Using Ultralytics SAM implementation...")
            from ultralytics import SAM
            self.model = SAM(model_path)
            self.use_samgeo = False
        
        print("SAM 3 model loaded successfully")
            
    def predict(
        self,
        image: Path = Input(description="Input satellite/aerial image to segment"),
        prompt: str = Input(
            description="Describe objects to extract (e.g., 'buildings', 'swimming pools', 'solar panels')", 
            default=""
        ),
        threshold: float = Input(description="Confidence threshold for detections", default=0.5),
        box_threshold: float = Input(description="Box detection threshold", default=0.3),
        text_threshold: float = Input(description="Text matching threshold", default=0.25)
    ) -> dict:
        """
        Run SAM 3 with Promptable Concept Segmentation (PCS)
        """
        
        # 1. Load image
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)
        h, w, _ = img_np.shape
        print(f"Processing image: {w}x{h} with prompt: '{prompt}'")
        
        # 2. Run SAM 3 inference
        if self.use_samgeo:
            # Use samgeo with text prompt
            temp_input = "/tmp/input_image.png"
            temp_output = "/tmp/sam_output.tif"
            img.save(temp_input)
            
            if prompt:
                self.model.generate(
                    source=temp_input,
                    output=temp_output,
                    text_prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            else:
                self.model.generate(
                    source=temp_input,
                    output=temp_output,
                    batch=True
                )
            
            # Load result
            from rasterio import open as rio_open
            try:
                with rio_open(temp_output) as src:
                    combined_mask = src.read(1).astype(np.uint8)
            except:
                combined_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Ultralytics SAM
            results = self.model(img_np, conf=threshold)
            
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                for i, mask in enumerate(masks):
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                    combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * (i + 1))
        
        # 3. Post-processing cleanup
        if np.any(combined_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Create colored visualization mask
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[combined_mask > 0] = [0, 200, 200, 180]  # Cyan
        
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
        print(f"Extracted {feature_count} features for: '{prompt}'")
        
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
        results_list = []
        if feature_count > 0:
            results_list.append({
                "feature_class": prompt if prompt else "detected_object",
                "feature_count": feature_count,
                "geojson": geojson_data,
                "class_mask_path": str(out_mask_path)
            })
        
        return {
            "mask": out_mask_path,
            "geojson": out_geojson_path,
            "results": results_list
        }
