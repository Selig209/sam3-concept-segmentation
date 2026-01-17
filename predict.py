"""
SAM 3 (Segment Anything Model 3) Predictor for Replicate/Cog
Uses official facebook/sam3 from HuggingFace Transformers (November 2025)

SAM 3 Features:
- Native text prompt segmentation (no grounding model needed)
- 0.9B parameters, improved architecture
- Direct text-to-mask generation
"""

from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
import os
import json
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform
import cv2

class Predictor(BasePredictor):
    def setup(self):
        """Load SAM 3 model from HuggingFace"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        from transformers import Sam3Processor, Sam3Model
        
        # Load SAM 3 from HuggingFace
        model_id = "facebook/sam3"
        print(f"Loading {model_id}...")
        
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        print("SAM 3 loaded successfully")
            
    def predict(
        self,
        image: Path = Input(description="Input satellite/aerial image to segment"),
        prompt: str = Input(
            description="Describe objects to extract (e.g., 'buildings', 'swimming pools', 'solar panels')", 
            default=""
        ),
        threshold: float = Input(description="Confidence threshold for mask", default=0.5),
        multimask_output: bool = Input(description="Return multiple mask options", default=False)
    ) -> dict:
        """
        Run SAM 3 segmentation with native text prompts
        """
        
        # 1. Load image
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)
        h, w, _ = img_np.shape
        print(f"Processing image: {w}x{h}")
        
        if prompt:
            print(f"Text prompt: '{prompt}'")
        
        # 2. Process inputs with SAM 3 (native text prompt support)
        if prompt:
            # SAM 3 supports direct text prompts
            inputs = self.processor(
                images=img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Automatic mask generation without prompt
            inputs = self.processor(
                images=img,
                return_tensors="pt"
            ).to(self.device)
        
        # 3. Run SAM 3 inference
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)
        
        # 4. Post-process masks
        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )
        
        # Get the best mask
        if len(masks) > 0 and len(masks[0]) > 0:
            # Use IoU scores to select best mask
            if hasattr(outputs, 'iou_scores'):
                best_idx = outputs.iou_scores[0].argmax().item()
                mask = masks[0][best_idx].cpu().numpy()
            else:
                mask = masks[0][0].cpu().numpy()
            
            # Apply threshold
            combined_mask = (mask > threshold).astype(np.uint8)
        else:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 5. Post-processing cleanup
        if np.any(combined_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 6. Create colored visualization mask
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[combined_mask > 0] = [0, 200, 200, 180]  # Cyan
        
        # 7. Vectorize to GeoJSON using contours
        geojson_features = []
        
        if np.any(combined_mask):
            contours, _ = cv2.findContours(
                combined_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLIFY
            )
            
            for idx, contour in enumerate(contours):
                if len(contour) >= 3:
                    # Convert contour to polygon coordinates
                    coords = contour.squeeze().tolist()
                    if len(coords) >= 3:
                        # Close the polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        
                        geojson_features.append({
                            "type": "Feature",
                            "properties": {
                                "class": prompt if prompt else "detected_object",
                                "instance_id": idx + 1,
                                "confidence": float(threshold)
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [coords]
                            }
                        })
        
        feature_count = len(geojson_features)
        print(f"Extracted {feature_count} features for: '{prompt}'")
        
        # 8. Save outputs
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
