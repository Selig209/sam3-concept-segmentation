"""
Grounded SAM Predictor for Replicate/Cog
Combines GroundingDINO (for text-to-boxes) with SAM (for precise segmentation)

Based on opengeos/geoai GroundedSAM class architecture (MIT License).
https://github.com/opengeos/geoai

Features inspired by GeoAI:
1. Text-to-box detection using GroundingDINO
2. Precise segmentation using SAM
3. Tiling support for large satellite/aerial imagery
4. Size filtering to remove noise
5. GeoJSON output with confidence scores

MIT License acknowledgment for GeoAI patterns used.
"""

from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
import os
import json
import cv2
from typing import List, Tuple, Optional

class Predictor(BasePredictor):
    def setup(self):
        """Load GroundingDINO + SAM models (inspired by GeoAI GroundedSAM)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load GroundingDINO for text-to-box detection
        # GeoAI uses grounding-dino-tiny for speed, we use base for accuracy
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            print("Loading GroundingDINO...")
            # Options: grounding-dino-tiny (faster) or grounding-dino-base (better)
            dino_model_id = "IDEA-Research/grounding-dino-base"
            self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
            self.dino_model.eval()
            print("GroundingDINO loaded successfully")
        except Exception as e:
            print(f"Error loading GroundingDINO: {e}")
            self.dino_model = None
        
        # Load SAM for precise segmentation
        try:
            from transformers import SamModel, SamProcessor
            
            print("Loading SAM...")
            sam_model_id = "facebook/sam-vit-huge"
            self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
            self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
            self.sam_model.eval()
            print("SAM loaded successfully")
        except Exception as e:
            print(f"Error loading SAM: {e}")
            self.sam_model = None
    
    def _generate_tiles(
        self, 
        width: int, 
        height: int, 
        tile_size: int = 1024, 
        overlap: int = 128
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate tile coordinates for large image processing.
        Inspired by GeoAI's tiling approach for satellite imagery.
        
        Args:
            width: Image width
            height: Image height
            tile_size: Size of each tile (default 1024 like GeoAI)
            overlap: Overlap between tiles to avoid edge artifacts (default 128)
        
        Returns:
            List of (x1, y1, x2, y2) tile coordinates
        """
        tiles = []
        step = tile_size - overlap
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                # Ensure minimum tile size
                if (x2 - x1) >= tile_size // 2 and (y2 - y1) >= tile_size // 2:
                    tiles.append((x1, y1, x2, y2))
        
        # If image is smaller than tile_size, use full image
        if not tiles:
            tiles.append((0, 0, width, height))
            
        return tiles
    
    def _process_tile(
        self, 
        tile_img: Image.Image, 
        prompt: str, 
        threshold: float,
        multimask_output: bool
    ) -> Tuple[np.ndarray, List[float], List[Tuple[float, float, float, float]]]:
        """
        Process a single tile with GroundingDINO + SAM.
        
        Returns:
            Tuple of (mask, scores, boxes)
        """
        h, w = tile_img.size[1], tile_img.size[0]
        tile_mask = np.zeros((h, w), dtype=np.uint8)
        scores = []
        boxes = []
        
        if self.dino_model is None or self.sam_model is None:
            return tile_mask, scores, boxes
        
        # GroundingDINO detection
        dino_inputs = self.dino_processor(
            images=tile_img,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            dino_outputs = self.dino_model(**dino_inputs)
        
        results = self.dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(h, w)]
        )[0]
        
        detected_boxes = results["boxes"].cpu().numpy()
        detected_scores = results["scores"].cpu().numpy()
        
        # SAM segmentation for each detected box
        for box, score in zip(detected_boxes, detected_scores):
            x1, y1, x2, y2 = box
            box_coords = [[x1, y1, x2, y2]]
            
            sam_inputs = self.sam_processor(
                images=tile_img,
                input_boxes=[box_coords],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs, multimask_output=multimask_output)
            
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu()
            )
            
            if len(masks) > 0 and len(masks[0]) > 0:
                if multimask_output and hasattr(sam_outputs, 'iou_scores'):
                    best_idx = sam_outputs.iou_scores[0][0].argmax().item()
                    mask = masks[0][0][best_idx].numpy()
                else:
                    mask = masks[0][0][0].numpy()
                
                tile_mask = np.maximum(tile_mask, (mask > 0.5).astype(np.uint8))
                scores.append(float(score))
                boxes.append((x1, y1, x2, y2))
        
        return tile_mask, scores, boxes
    
    def _merge_tile_masks(
        self,
        full_mask: np.ndarray,
        tile_mask: np.ndarray,
        tile_coords: Tuple[int, int, int, int],
        overlap: int
    ) -> np.ndarray:
        """
        Merge a tile mask into the full mask with overlap blending.
        Inspired by GeoAI's approach to avoid seam artifacts.
        """
        x1, y1, x2, y2 = tile_coords
        
        # Simple maximum merge for overlapping regions
        # This naturally handles the overlap without creating seams
        current_region = full_mask[y1:y2, x1:x2]
        full_mask[y1:y2, x1:x2] = np.maximum(current_region, tile_mask[:y2-y1, :x2-x1])
        
        return full_mask
            
    def predict(
        self,
        image: Path = Input(description="Input satellite/aerial image to segment"),
        prompt: str = Input(
            description="Describe objects to extract (e.g., 'buildings', 'swimming pools', 'solar panels')", 
            default=""
        ),
        threshold: float = Input(description="Confidence threshold for detection", default=0.3),
        multimask_output: bool = Input(description="Return multiple mask options", default=False),
        use_tiling: bool = Input(
            description="Use tiling for large images (recommended for >2048px)", 
            default=True
        ),
        tile_size: int = Input(
            description="Tile size in pixels (like GeoAI default: 1024)",
            default=1024,
            ge=256,
            le=2048
        ),
        tile_overlap: int = Input(
            description="Overlap between tiles in pixels (reduces edge artifacts)",
            default=128,
            ge=0,
            le=512
        ),
        min_mask_area: int = Input(
            description="Minimum mask area in pixels (filters noise)",
            default=100,
            ge=0
        )
    ) -> dict:
        """
        Run GroundingDINO + SAM segmentation with text prompts.
        Supports tiling for large satellite/aerial imagery.
        
        Processing steps:
        1. Split image into overlapping tiles (if enabled)
        2. GroundingDINO detects objects from text description per tile
        3. SAM creates precise masks from the detected boxes
        4. Merge tile masks with overlap handling
        5. Filter small regions and vectorize to GeoJSON
        """
        
        # 1. Load image
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)
        h, w, _ = img_np.shape
        print(f"Processing image: {w}x{h}")
        
        if not prompt:
            prompt = "object"
        print(f"Text prompt: '{prompt}'")
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        all_scores = []
        
        # 2. Decide whether to use tiling
        should_tile = use_tiling and (w > tile_size or h > tile_size)
        
        if should_tile:
            # Generate tiles for large image
            tiles = self._generate_tiles(w, h, tile_size, tile_overlap)
            print(f"Processing {len(tiles)} tiles ({tile_size}x{tile_size} with {tile_overlap}px overlap)")
            
            for i, (x1, y1, x2, y2) in enumerate(tiles):
                # Extract tile
                tile_img = img.crop((x1, y1, x2, y2))
                print(f"  Tile {i+1}/{len(tiles)}: ({x1},{y1}) to ({x2},{y2})")
                
                # Process tile
                tile_mask, scores, boxes = self._process_tile(
                    tile_img, prompt, threshold, multimask_output
                )
                
                # Merge into full mask
                combined_mask = self._merge_tile_masks(
                    combined_mask, tile_mask, (x1, y1, x2, y2), tile_overlap
                )
                all_scores.extend(scores)
                
                print(f"    Found {len(boxes)} objects in tile")
        else:
            # Process entire image at once (for smaller images)
            print("Processing full image (no tiling)")
            
            if self.dino_model is not None and self.sam_model is not None:
                # Process with GroundingDINO
                dino_inputs = self.dino_processor(
                    images=img,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    dino_outputs = self.dino_model(**dino_inputs)
                
                results = self.dino_processor.post_process_grounded_object_detection(
                    dino_outputs,
                    dino_inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=[(h, w)]
                )[0]
                
                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                
                print(f"GroundingDINO found {len(boxes)} objects")
                
                # SAM segmentation for each box
                for idx, (box, score) in enumerate(zip(boxes, scores)):
                    x1, y1, x2, y2 = box
                    box_coords = [[x1, y1, x2, y2]]
                    
                    sam_inputs = self.sam_processor(
                        images=img,
                        input_boxes=[box_coords],
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        sam_outputs = self.sam_model(**sam_inputs, multimask_output=multimask_output)
                    
                    masks = self.sam_processor.image_processor.post_process_masks(
                        sam_outputs.pred_masks.cpu(),
                        sam_inputs["original_sizes"].cpu(),
                        sam_inputs["reshaped_input_sizes"].cpu()
                    )
                    
                    if len(masks) > 0 and len(masks[0]) > 0:
                        if multimask_output and hasattr(sam_outputs, 'iou_scores'):
                            best_idx = sam_outputs.iou_scores[0][0].argmax().item()
                            mask = masks[0][0][best_idx].numpy()
                        else:
                            mask = masks[0][0][0].numpy()
                        
                        combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8))
                        all_scores.append(float(score))
                        print(f"  Object {idx+1}: score={score:.2f}")
            else:
                print("Warning: Models not loaded properly")
        
        # 3. Post-processing cleanup
        if np.any(combined_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Create colored visualization mask
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[combined_mask > 0] = [0, 200, 200, 180]  # Cyan
        
        # 5. Vectorize to GeoJSON using contours with size filtering
        geojson_features = []
        
        if np.any(combined_mask):
            contours, _ = cv2.findContours(
                combined_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for idx, contour in enumerate(contours):
                # Filter by minimum area (inspired by GeoAI's min_mask_size)
                area = cv2.contourArea(contour)
                if area < min_mask_area:
                    continue
                    
                if len(contour) >= 3:
                    # Simplify contour
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert contour to polygon coordinates
                    coords = approx.squeeze().tolist()
                    if isinstance(coords[0], int):
                        coords = [coords]
                    if len(coords) >= 3:
                        # Close the polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        
                        avg_score = np.mean(all_scores) if all_scores else threshold
                        geojson_features.append({
                            "type": "Feature",
                            "properties": {
                                "class": prompt,
                                "instance_id": len(geojson_features) + 1,
                                "confidence": float(avg_score),
                                "area_pixels": int(area)
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [coords]
                            }
                        })
        
        feature_count = len(geojson_features)
        tile_info = f" ({len(tiles)} tiles)" if should_tile else ""
        print(f"Extracted {feature_count} features for: '{prompt}'{tile_info}")
        
        # 6. Save outputs
        out_mask_path = Path("/tmp/sam3_mask.png")
        Image.fromarray(mask_rgba).save(str(out_mask_path))
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_features,
            "properties": {
                "prompt": prompt,
                "threshold": threshold,
                "tile_size": tile_size if should_tile else None,
                "tile_overlap": tile_overlap if should_tile else None,
                "tiles_processed": len(tiles) if should_tile else 1,
                "min_mask_area": min_mask_area
            }
        }
        out_geojson_path = Path("/tmp/sam3_output.geojson")
        with open(str(out_geojson_path), "w") as f:
            json.dump(geojson_data, f)
        
        # Return results
        results_list = []
        if feature_count > 0:
            results_list.append({
                "feature_class": prompt,
                "feature_count": feature_count,
                "geojson": geojson_data,
                "class_mask_path": str(out_mask_path)
            })
        
        return {
            "mask": out_mask_path,
            "geojson": out_geojson_path,
            "results": results_list
        }
