import base64
import cv2
import math
from copy import deepcopy
import numpy as np
from scipy.ndimage import label, distance_transform_edt
from scipy.signal import find_peaks
# python -m pip install -U scikit-image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class ViewSeparator:
    """
    Separate multiple engineering views from a single blueprint image.
    Handles cases where views are close together or slightly overlapping.
    """
    
    def __init__(self,
                 min_area_ratio=0.02,
                 max_area_ratio=0.3,
                 max_aspect_ratio=20,
                 min_aspect_ratio=0.1,
                 watershed_min_distance=50,
                 watershed_threshold=0.3,
                 projection_cut_threshold=0.1,
                 edge_margin_ratio=0.03,
                 primary_gap_threshold=0.05,
                 primary_gap_min_ratio=0.08,
                 secondary_gap_min_ratio=0.04,
                 merge_distance_ratio=0.0005,
                 merge_area_ratio=0.05):
        """
        Initialize ViewSeparator with configurable parameters.
        
        Args:
            min_area_ratio: Minimum view area as fraction of image (0.02 = 2%)
            max_area_ratio: Maximum area before considered suspicious (0.3 = 30%)
            max_aspect_ratio: Maximum width/height ratio before suspicious
            min_aspect_ratio: Minimum width/height ratio before suspicious
            watershed_min_distance: Minimum distance between view centers (pixels)
            watershed_threshold: Threshold for distance transform in watershed
            projection_cut_threshold: Threshold for projection-based cutting (0.1 = 10%)
            edge_margin_ratio: Margin from edges to exclude (0.03 = 3%)
            primary_gap_threshold: Threshold for identifying dominant blank gaps
            primary_gap_min_ratio: Minimum relative size of dominant gap vs. image dimension
            secondary_gap_min_ratio: Minimum relative size of secondary gaps used for splitting
            merge_distance_ratio: Max normalized distance (fraction of image diagonal) to merge nearby small views
            merge_area_ratio: Max area ratio (vs image) for views considered mergeable
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.watershed_min_distance = watershed_min_distance
        self.watershed_threshold = watershed_threshold
        self.projection_cut_threshold = projection_cut_threshold
        self.edge_margin_ratio = edge_margin_ratio
        self.primary_gap_threshold = primary_gap_threshold
        self.primary_gap_min_ratio = primary_gap_min_ratio
        self.secondary_gap_min_ratio = secondary_gap_min_ratio
        self.merge_distance_ratio = merge_distance_ratio
        self.merge_area_ratio = merge_area_ratio
        self._dividers = []
        self.last_run_outputs = {}
    
    def _load_and_resize_image_for_frames(self, image, target_size=3840, enhance_lines=True):
        """
        Optimized for ship blueprint frame detection
        - Preserves thin lines
        - Optional line enhancement (thickening)
        - Uses appropriate interpolation
        - Maintains aspect ratio
        
        Args:
            image: Can be either:
                   - str: image file path
                   - numpy.ndarray: image data (BGR format)
        """
        # Handle file path (string)
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
        
        # Handle PIL Image (check for PIL-specific attributes)
        elif hasattr(image, 'mode') and hasattr(image, 'size'):
            # Convert PIL Image to numpy array, then RGB to BGR
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Handle numpy array
        elif hasattr(image, 'shape') and len(image.shape) in [2, 3]:
            # If grayscale (2D), convert to BGR
            if len(image.shape) == 2:
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img = image.copy()  # Use provided image data directly
        
        # Invalid input
        else:
            raise ValueError(
                f"Invalid image input. Expected file path (str), "
                f"PIL.Image.Image, or image array (numpy.ndarray), "
                f"got {type(image)}"
            )
            
        orig_h, orig_w = img.shape[:2]

        # Resize logic (same as before)
        long_side = max(orig_w, orig_h)

        if target_size is None or target_size <= 0 or long_side <= target_size:
            scale = 1.0
            processed = img
        else:
            scale = float(target_size) / float(long_side)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            processed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Line enhancement (thickening)
        if enhance_lines:
            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Detect edges (lines in blueprint)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to thicken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # 2,2 # Adjust size for thickness
            thickened_edges = cv2.dilate(edges, kernel, iterations=2)  # 2
            
            # Apply thickened edges back to image
            # Make lines darker/more prominent
            processed[thickened_edges > 0] = [0, 0, 0]  # Make lines black
            
            # Alternative: Morphological closing to fill gaps in lines
            # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close)

        proc_h, proc_w = processed.shape[:2]

        resize_info = {
            'orig_shape': (orig_h, orig_w),
            'processed_shape': (proc_h, proc_w),
            'scale': scale,
            'resized': scale != 1.0,
            'target_size': target_size,
            'enhanced': enhance_lines
        }

        return img, processed, resize_info

    def _load_output_image(self, image):
        """Load visualization/output image without resizing or enhancement."""
        if image is None:
            raise ValueError("Visualization image is None")

        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load visualization image from path: {image}")
            return img
        elif hasattr(image, 'mode') and hasattr(image, 'size'):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif hasattr(image, 'shape') and len(image.shape) in [2, 3]:
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image.copy()

        raise ValueError(
            f"Invalid visualization image input. Expected file path (str), "
            f"PIL.Image.Image, or image array (numpy.ndarray), got {type(image)}"
        )

    def _scale_and_clip_bbox(self, bbox, scale_x, scale_y, max_width, max_height):
        """Scale bbox coordinates and clip to image bounds."""
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        scaled = [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y))
        ]

        return self._clip_bbox(scaled, max_width, max_height)

    def _clip_bbox(self, bbox, max_width, max_height):
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, max_width - 1))
        y1 = max(0, min(y1, max_height - 1))
        x2 = max(0, min(x2, max_width))
        y2 = max(0, min(y2, max_height))

        if x2 < x1:
            x2 = x1
        if y2 < y1:
            y2 = y1

        return [x1, y1, x2, y2]

    def _scale_dividers(self, dividers, scale_x, scale_y, max_width, max_height):
        if not dividers:
            return []

        scaled_dividers = []
        for divider in dividers:
            if isinstance(divider, dict):
                bbox = divider.get('bbox')
                scaled_bbox = self._scale_and_clip_bbox(bbox, scale_x, scale_y, max_width, max_height) if bbox else None
                new_divider = dict(divider)
                if scaled_bbox:
                    new_divider['bbox'] = scaled_bbox
                scaled_dividers.append(new_divider)
            else:
                scaled_bbox = self._scale_and_clip_bbox(divider, scale_x, scale_y, max_width, max_height)
                scaled_dividers.append(scaled_bbox)

        return scaled_dividers

    def _encode_image_to_base64(self, image, encode_format='jpeg'):
        if image is None:
            return None

        format_lower = (encode_format or 'jpeg').lower()
        if format_lower in ('jpg', 'jpeg'):
            ext = '.jpg'
        elif format_lower in ('png',):
            ext = '.png'
        else:
            raise ValueError(f"Unsupported encode_format: {encode_format}. Use 'jpeg' or 'png'.")

        success, buffer = cv2.imencode(ext, image)
        if not success:
            raise ValueError("Failed to encode image for base64 output")

        return base64.b64encode(buffer).decode('utf-8')

    def _expand_bbox_with_gaps(self, bbox, dividers, img_width, img_height, gap_threshold):
        if bbox is None or not dividers:
            return bbox

        x1, y1, x2, y2 = bbox

        for divider in dividers:
            if divider is None:
                continue

            if isinstance(divider, dict):
                div_bbox = divider.get('bbox')
                orientation = divider.get('orientation')
            else:
                div_bbox = divider
                orientation = None

            if not div_bbox:
                continue

            dx1, dy1, dx2, dy2 = div_bbox

            if orientation == 'vertical':
                if dy1 < y2 and dy2 > y1:
                    if (0 <= x1 - dx2 <= gap_threshold) or (0 <= dx1 - x2 <= gap_threshold) or (dx1 <= x2 and dx2 >= x1):
                        x1 = min(x1, dx1)
                        x2 = max(x2, dx2)
            elif orientation == 'horizontal':
                if dx1 < x2 and dx2 > x1:
                    if (0 <= y1 - dy2 <= gap_threshold) or (0 <= dy1 - y2 <= gap_threshold) or (dy1 <= y2 and dy2 >= y1):
                        y1 = min(y1, dy1)
                        y2 = max(y2, dy2)
            else:
                if dx1 < x2 and dx2 > x1 and dy1 < y2 and dy2 > y1:
                    x1 = min(x1, dx1)
                    y1 = min(y1, dy1)
                    x2 = max(x2, dx2)
                    y2 = max(y2, dy2)

        return self._clip_bbox([x1, y1, x2, y2], img_width, img_height)

    def separate_views(self,
                       image,
                       visualization_image=None,
                       visualize=True,
                       save_path='view_separation_result.jpg',
                       return_outputs=False,
                       encode_format='jpeg'):
        """
        Main method to separate views from blueprint image.
        
        Args:
            image: Input image (file path or numpy array) used for processing.
            visualization_image: Optional second image (path or ndarray) used for visualization/cropping outputs.
            visualize: Whether to save visualization
            save_path: Optional path to save visualization (applies to visualization_image if provided)
            return_outputs: When True, returns a dictionary containing views and output artifacts.
            encode_format: Output image encoding for base64 payloads (default JPEG).
            
        Returns:
            List of view dictionaries (default) or a dictionary with views and artifacts when return_outputs=True
        """
        # Load and preprocess image
        original_img, processed_img, resize_info = self._load_and_resize_image_for_frames(image)
        self._dividers = []
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        img_area = height * width
        
        print(f"Processing image: {width}x{height} (area: {img_area})")
        
        # Step 1: Binarization (without dilation to avoid merging)
        binary = self._binarize_image(gray)
        self._dividers = self._detect_primary_gap(binary, width, height)
        
        # Step 2: Find all candidate regions
        candidates = self._find_candidate_regions(binary, img_area)
        print(f"Found {len(candidates)} candidate regions")
        
        if not candidates:
            return []
        
        # Step 3: Filter edge regions
        candidates = self._filter_edge_regions(candidates, width, height)
        print(f"After edge filtering: {len(candidates)} regions")
        
        if not candidates:
            return []
        
        # Step 4: Separate suspicious regions (potentially merged views)
        all_views = []
        
        for candidate in candidates:
            if self._is_suspicious(candidate, img_area):
                print(f"Suspicious region detected: {candidate['bbox']}")
                # Try to separate this region
                sub_views = self._separate_merged_region(candidate, binary, img_area)
                all_views.extend(sub_views)
            else:
                # Simple region, accept as-is
                all_views.append(candidate)
        
        print(f"Total views after separation: {len(all_views)}")
        
        # Step 5: Merge small/fragmented views based on gaps and proximity
        merged_views = self._merge_views(all_views, width, height)
        if len(merged_views) != len(all_views):
            print(f"Views after merging: {len(merged_views)} (from {len(all_views)})")

        # Step 6: Post-process and add metadata
        final_views = self._finalize_views(merged_views, width, height)
        
        # Prepare visualization image (second image or original)
        output_img = self._load_output_image(visualization_image) if visualization_image is not None else original_img.copy()
        output_height, output_width = output_img.shape[:2]

        # Compute scaling from processing space to output space
        scale_x = output_width / max(width, 1)
        scale_y = output_height / max(height, 1)
        scaled_dividers = self._scale_dividers(self._dividers, scale_x, scale_y, output_width, output_height)
        gap_threshold = max(5, int(round(min(output_width, output_height) * 0.005)))

        # Update view coordinates for output space
        for view in final_views:
            x1, y1, x2, y2 = view['bbox']
            bbox_output = self._scale_and_clip_bbox([x1, y1, x2, y2], scale_x, scale_y, output_width, output_height)
            if view.get('method') == 'merged':
                expanded_bbox = self._expand_bbox_with_gaps(bbox_output, scaled_dividers, output_width, output_height, gap_threshold)
                if expanded_bbox:
                    bbox_output = expanded_bbox
            view['bbox_output'] = bbox_output
            ox1, oy1, ox2, oy2 = bbox_output
            view['bbox_norm_output'] = [
                ox1 / max(output_width, 1),
                oy1 / max(output_height, 1),
                ox2 / max(output_width, 1),
                oy2 / max(output_height, 1)
            ]
            ocx = (ox1 + ox2) / 2.0 / max(output_width, 1)
            ocy = (oy1 + oy2) / 2.0 / max(output_height, 1)
            ow = (ox2 - ox1) / max(output_width, 1)
            oh = (oy2 - oy1) / max(output_height, 1)
            view['bbox_yolo_output'] = {'cx': ocx, 'cy': ocy, 'w': ow, 'h': oh}

        # Generate cropped outputs
        view_images = []

        for view in final_views:
            if 'bbox_output' not in view:
                continue
            ox1, oy1, ox2, oy2 = view['bbox_output']
            if ox2 - ox1 <= 0 or oy2 - oy1 <= 0:
                continue
            crop = output_img[oy1:oy2, ox1:ox2].copy()
            image_base64 = self._encode_image_to_base64(crop, encode_format)
            crop_record = {
                'id': view['id'],
                'bbox': view['bbox_output'],
                'image_base64': image_base64
            }
            view['image_base64'] = image_base64
            view_images.append(crop_record)

        # Step 7: Visualize if requested (on output image coordinates)
        visualization_image_array = None
        visualization_path = None
        if visualize and final_views:
            visualization_image_array = self._visualize_results(
                output_img,
                final_views,
                save_path,
                scaled_dividers,
                bbox_key='bbox_output'
            )
            visualization_path = save_path

        base_image_for_encoding = visualization_image_array if visualization_image_array is not None else output_img
        full_image_base64 = self._encode_image_to_base64(base_image_for_encoding, encode_format)

        self.last_run_outputs = {
            'views': final_views,
            'resize_info': resize_info,
            'visualization_path': visualization_path,
            'visualization_image': visualization_image_array,
            'view_images': view_images,
            'full_image_base64': full_image_base64,
            'scale_to_output': {'x': scale_x, 'y': scale_y},
            'scaled_dividers': scaled_dividers,
            'output_shape': (output_height, output_width),
            'processing_shape': (height, width)
        }

        if return_outputs:
            return self.last_run_outputs

        return final_views
    
    def _binarize_image(self, gray):
        """
        Convert grayscale to binary.
        Uses Otsu's method for automatic thresholding.
        """
        # Denoise first
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's binarization
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Light cleanup (remove very small noise)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary

    def _detect_primary_gap(self, binary, img_width, img_height):
        """
        Identify the dominant blank region (gap) across the entire image.

        Returns a list with at most one divider describing the largest gap.
        """
        # Normalize projections to [0, 1]
        h_projection = np.sum(binary, axis=1) / max(img_width, 1) / 255.0
        v_projection = np.sum(binary, axis=0) / max(img_height, 1) / 255.0

        horizontal_gaps = self._find_gap_segments(h_projection, self.primary_gap_threshold)
        vertical_gaps = self._find_gap_segments(v_projection, self.primary_gap_threshold)

        candidates = []

        if horizontal_gaps:
            best_h_gap = max(horizontal_gaps, key=lambda g: g['length'])
            if best_h_gap['ratio'] >= self.primary_gap_min_ratio:
                y1 = int(best_h_gap['start'])
                y2 = int(best_h_gap['end'])
                area = max(1, y2 - y1) * img_width
                candidates.append({
                    'area': area,
                    'divider': {
                        'bbox': [0, y1, img_width, y2],
                        'orientation': 'horizontal',
                        'type': 'primary',
                        'ratio': best_h_gap['ratio']
                    }
                })

        if vertical_gaps:
            best_v_gap = max(vertical_gaps, key=lambda g: g['length'])
            if best_v_gap['ratio'] >= self.primary_gap_min_ratio:
                x1 = int(best_v_gap['start'])
                x2 = int(best_v_gap['end'])
                area = max(1, x2 - x1) * img_height
                candidates.append({
                    'area': area,
                    'divider': {
                        'bbox': [x1, 0, x2, img_height],
                        'orientation': 'vertical',
                        'type': 'primary',
                        'ratio': best_v_gap['ratio']
                    }
                })

        if not candidates:
            return []

        # Return divider with largest blank area (dominant separation)
        best_candidate = max(candidates, key=lambda c: c['area'])
        return [best_candidate['divider']]

    def _find_gap_segments(self, projection, threshold):
        """
        Locate contiguous low-density regions (gaps) in a 1D projection profile.
        Returns a list of dictionaries with start, end, center, length, and ratio.
        """
        if projection.size == 0:
            return []

        below_threshold = projection < threshold
        if not np.any(below_threshold):
            return []

        segments = []
        start = None

        for idx, is_low in enumerate(below_threshold):
            if is_low and start is None:
                start = idx
            elif not is_low and start is not None:
                end = idx
                length = max(1, end - start)
                segments.append({
                    'start': start,
                    'end': end,
                    'center': start + length // 2,
                    'length': length,
                    'ratio': length / projection.size
                })
                start = None

        # Handle trailing gap
        if start is not None:
            end = projection.size
            length = max(1, end - start)
            segments.append({
                'start': start,
                'end': end,
                'center': start + length // 2,
                'length': length,
                'ratio': length / projection.size
            })

        return segments
    
    def _find_candidate_regions(self, binary, img_area):
        """
        Find all candidate regions using connected component analysis.
        """
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        min_area = img_area * self.min_area_ratio
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Basic validation
            if w < 10 or h < 10:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            
            candidate = {
                'id': i,
                'contour': contour,
                'bbox': [x, y, x + w, y + h],
                'area': area,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def _filter_edge_regions(self, candidates, img_width, img_height):
        """
        Remove regions too close to image edges (likely borders or watermarks).
        """
        margin_x = img_width * self.edge_margin_ratio
        margin_y = img_height * self.edge_margin_ratio
        
        filtered = []
        
        for candidate in candidates:
            x1, y1, x2, y2 = candidate['bbox']
            
            # Check if too close to any edge
            too_close = (
                x1 < margin_x or
                y1 < margin_y or
                x2 > img_width - margin_x or
                y2 > img_height - margin_y
            )
            
            if not too_close:
                filtered.append(candidate)
        
        return filtered
    
    def _is_suspicious(self, candidate, img_area):
        """
        Determine if a region is suspicious (potentially merged views).
        
        Criteria:
        1. Area too large (> max_area_ratio of image)
        2. Aspect ratio abnormal (too wide or too tall)
        3. Complexity too high (perimeter^2 / area)
        """
        area = candidate['area']
        aspect = candidate['aspect_ratio']
        
        # Check area
        if area > img_area * self.max_area_ratio:
            return True
        
        # Check aspect ratio
        if aspect > self.max_aspect_ratio or aspect < self.min_aspect_ratio:
            return True
        
        # Check complexity (perimeter vs area)
        contour = candidate['contour']
        perimeter = cv2.arcLength(contour, True)
        if area > 0:
            complexity = (perimeter ** 2) / area
            # Empirical threshold: simple rectangle has complexity ~16
            # Complex merged shapes have higher values
            if complexity > 50:
                return True
        
        return False
    
    def _separate_merged_region(self, candidate, binary, img_area):
        """
        Separate a suspicious region that may contain multiple merged views.
        
        Strategy:
        1. Try projection-based cutting first (fast, works for horizontal/vertical merging)
        2. If projection fails, use watershed (handles complex merging)
        """
        x1, y1, x2, y2 = candidate['bbox']
        region_mask = binary[y1:y2, x1:x2].copy()
        
        # Try projection-based separation
        projection_views, projection_dividers = self._separate_by_projection(region_mask, x1, y1)
        
        if projection_dividers:
            self._dividers.extend(projection_dividers)
        
        if len(projection_views) > 1:
            print(f"  -> Separated by projection into {len(projection_views)} views")
            return projection_views
        
        # If projection fails, try watershed
        watershed_views = self._separate_by_watershed(region_mask, x1, y1)
        
        if len(watershed_views) > 1:
            print(f"  -> Separated by watershed into {len(watershed_views)} views")
            return watershed_views
        
        # If all methods fail, return original region
        print(f"  -> Could not separate, keeping original")
        return [candidate]
    
    def _separate_by_projection(self, region_mask, offset_x, offset_y):
        """
        Separate region by finding gaps in projection histogram.
        """
        height, width = region_mask.shape
        
        # Compute horizontal projection (sum along columns)
        h_projection = np.sum(region_mask, axis=1) / 255.0
        
        # Compute vertical projection (sum along rows)
        v_projection = np.sum(region_mask, axis=0) / 255.0
        
        # Normalize projections
        h_projection = h_projection / width if width > 0 else h_projection
        v_projection = v_projection / height if height > 0 else v_projection
        
        # Find gaps (low projection values)
        h_gaps = self._find_projection_gaps(
            h_projection,
            self.projection_cut_threshold,
            self.secondary_gap_min_ratio
        )
        v_gaps = self._find_projection_gaps(
            v_projection,
            self.projection_cut_threshold,
            self.secondary_gap_min_ratio
        )
        
        # Choose dominant direction (more gaps = better separation)
        if len(h_gaps) > len(v_gaps) and len(h_gaps) > 0:
            # Horizontal gaps -> cut horizontally (separate top/bottom views)
            return self._cut_by_horizontal_gaps(region_mask, h_gaps, offset_x, offset_y)
        elif len(v_gaps) > 0:
            # Vertical gaps -> cut vertically (separate left/right views)
            return self._cut_by_vertical_gaps(region_mask, v_gaps, offset_x, offset_y)
        
        # No clear gaps found
        return [], []
    
    def _find_projection_gaps(self, projection, threshold, min_ratio):
        """
        Find gaps in projection histogram where values are below threshold.
        Returns list of gap segment dictionaries.
        """
        segments = self._find_gap_segments(projection, threshold)
        return [segment for segment in segments if segment['ratio'] >= min_ratio]
    
    def _cut_by_horizontal_gaps(self, region_mask, gaps, offset_x, offset_y):
        """
        Cut region horizontally at gap positions.
        """
        height, width = region_mask.shape
        views = []
        dividers = []
        
        # Sort gaps and prepare boundaries
        gaps = sorted(gaps, key=lambda g: g['center'])
        gap_centers = [int(g['center']) for g in gaps]
        boundaries = [0] + gap_centers + [height]
        
        # Extract sub-regions
        for i in range(len(boundaries) - 1):
            y_start = boundaries[i]
            y_end = boundaries[i + 1]

            # Skip if region too small
            if y_end - y_start < 10:
                continue

            sub_region = region_mask[y_start:y_end, :]

            # Find actual content bounds in sub-region
            contours, _ = cv2.findContours(sub_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                sx, sy, sw, sh = cv2.boundingRect(largest)

                view = {
                    'bbox': [offset_x + sx, offset_y + y_start + sy,
                             offset_x + sx + sw, offset_y + y_start + sy + sh],
                    'area': sw * sh,
                    'width': sw,
                    'height': sh,
                    'aspect_ratio': sw / sh if sh > 0 else 0,
                    'method': 'projection_horizontal'
                }
                views.append(view)

        # Create dividers at gap positions using actual gap bounds
        for gap in gaps:
            y1 = np.clip(int(gap['start']), 0, height)
            y2 = np.clip(int(gap['end']), 0, height)
            if y2 - y1 <= 0:
                continue
            dividers.append({
                'bbox': [offset_x, offset_y + y1, offset_x + width, offset_y + y2],
                'orientation': 'horizontal',
                'type': 'secondary',
                'ratio': gap['ratio']
            })

        return views, dividers
    
    def _cut_by_vertical_gaps(self, region_mask, gaps, offset_x, offset_y):
        """
        Cut region vertically at gap positions.
        """
        height, width = region_mask.shape
        views = []
        dividers = []
        
        # Sort gaps and prepare boundaries
        gaps = sorted(gaps, key=lambda g: g['center'])
        gap_centers = [int(g['center']) for g in gaps]
        boundaries = [0] + gap_centers + [width]
        
        # Extract sub-regions
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            
            # Skip if region too small
            if x_end - x_start < 10:
                continue
            
            sub_region = region_mask[:, x_start:x_end]
            
            # Find actual content bounds in sub-region
            contours, _ = cv2.findContours(sub_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                sx, sy, sw, sh = cv2.boundingRect(largest)
                
                view = {
                    'bbox': [offset_x + x_start + sx, offset_y + sy,
                            offset_x + x_start + sx + sw, offset_y + sy + sh],
                    'area': sw * sh,
                    'width': sw,
                    'height': sh,
                    'aspect_ratio': sw / sh if sh > 0 else 0,
                    'method': 'projection_vertical'
                }
                views.append(view)
        
        # Create dividers at gap positions using actual gap bounds
        for gap in gaps:
            x1 = np.clip(int(gap['start']), 0, width)
            x2 = np.clip(int(gap['end']), 0, width)
            if x2 - x1 <= 0:
                continue
            dividers.append({
                'bbox': [offset_x + x1, offset_y, offset_x + x2, offset_y + height],
                'orientation': 'vertical',
                'type': 'secondary',
                'ratio': gap['ratio']
            })
        
        return views, dividers
    
    def _separate_by_watershed(self, region_mask, offset_x, offset_y):
        """
        Separate region using watershed algorithm.
        Finds view centers using distance transform, then grows regions from centers.
        """
        # Distance transform: find pixels far from edges (view centers)
        dist_transform = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
        
        # Normalize
        dist_transform = dist_transform / dist_transform.max() if dist_transform.max() > 0 else dist_transform
        
        # Find local maxima (view centers)
        # Use threshold to filter weak peaks
        threshold_value = self.watershed_threshold
        local_max_coords = peak_local_max(
            dist_transform,
            min_distance=self.watershed_min_distance,
            threshold_abs=threshold_value
        )
        
        if len(local_max_coords) < 2:
            # Not enough peaks, cannot separate
            return []
        
        print(f"  -> Found {len(local_max_coords)} view centers")
        
        # Create markers for watershed
        markers = np.zeros_like(region_mask, dtype=np.int32)
        for i, (y, x) in enumerate(local_max_coords):
            markers[y, x] = i + 1
        
        # Expand markers slightly to avoid single-pixel markers
        markers = cv2.dilate(markers.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        markers = markers.astype(np.int32)
        
        # Apply watershed
        # Note: watershed expects negative distance transform (valleys become peaks)
        labels = watershed(-dist_transform, markers, mask=region_mask)
        
        # Extract regions from watershed result
        views = []
        unique_labels = np.unique(labels)
        
        for label_id in unique_labels:
            if label_id == 0:  # Background
                continue
            
            # Create mask for this region
            region = (labels == label_id).astype(np.uint8) * 255
            
            # Find bounding box
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                
                view = {
                    'bbox': [offset_x + x, offset_y + y, offset_x + x + w, offset_y + y + h],
                    'area': w * h,
                    'width': w,
                    'height': h,
                    'aspect_ratio': w / h if h > 0 else 0,
                    'method': 'watershed'
                }
                views.append(view)
        
        return views
    
    def _merge_views(self, views, img_width, img_height):
        """
        Merge fragmented or small views based on gap dividers and spatial proximity.

        Strategy:
        1. Treat primary/secondary gaps as hard separators.
        2. Consider merging views when at least one is small and they are close.
        3. Union connected components to form larger views.
        """
        if len(views) <= 1:
            return views

        img_area = img_width * img_height
        mergeable_area = img_area * self.merge_area_ratio
        max_distance = math.hypot(img_width, img_height) * self.merge_distance_ratio

        parents = list(range(len(views)))

        def find(idx):
            while parents[idx] != idx:
                parents[idx] = parents[parents[idx]]
                idx = parents[idx]
            return idx

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parents[root_j] = root_i

        def is_small(view):
            return view.get('area', 0) <= mergeable_area

        def bbox_distance(view_a, view_b):
            ax1, ay1, ax2, ay2 = view_a['bbox']
            bx1, by1, bx2, by2 = view_b['bbox']
            dx = max(0, max(ax1 - bx2, bx1 - ax2))
            dy = max(0, max(ay1 - by2, by1 - ay2))
            return math.hypot(dx, dy)

        def ranges_overlap(range_a, range_b):
            return range_a[0] < range_b[1] and range_b[0] < range_a[1]

        def divider_blocks(view_a, view_b, divider):
            if not isinstance(divider, dict):
                return False

            x1, y1, x2, y2 = divider['bbox']
            orientation = divider.get('orientation')

            ax1, ay1, ax2, ay2 = view_a['bbox']
            bx1, by1, bx2, by2 = view_b['bbox']

            if orientation == 'horizontal':
                above_gap = ay2 <= y1 and by1 >= y2
                below_gap = by2 <= y1 and ay1 >= y2
                if (above_gap or below_gap):
                    combined_x = (min(ax1, bx1), max(ax2, bx2))
                    if ranges_overlap(combined_x, (x1, x2)):
                        return True
            elif orientation == 'vertical':
                left_gap = ax2 <= x1 and bx1 >= x2
                right_gap = bx2 <= x1 and ax1 >= x2
                if (left_gap or right_gap):
                    combined_y = (min(ay1, by1), max(ay2, by2))
                    if ranges_overlap(combined_y, (y1, y2)):
                        return True

            return False

        def separated_by_gap(view_a, view_b):
            for divider in self._dividers:
                if divider_blocks(view_a, view_b, divider):
                    return True
            return False

        view_count = len(views)
        for i in range(view_count):
            for j in range(i + 1, view_count):
                if not (is_small(views[i]) or is_small(views[j])):
                    continue
                if separated_by_gap(views[i], views[j]):
                    continue
                distance = bbox_distance(views[i], views[j])
                if distance <= max_distance:
                    union(i, j)

        clusters = {}
        for idx, view in enumerate(views):
            root = find(idx)
            clusters.setdefault(root, []).append(view)

        merged_views = []
        for cluster_views in clusters.values():
            if len(cluster_views) == 1:
                merged_views.append(cluster_views[0])
                continue

            xs1, ys1, xs2, ys2 = zip(*(v['bbox'] for v in cluster_views))
            merged_bbox = [min(xs1), min(ys1), max(xs2), max(ys2)]
            width = merged_bbox[2] - merged_bbox[0]
            height = merged_bbox[3] - merged_bbox[1]
            area = width * height

            merged_view = {
                'bbox': merged_bbox,
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'method': 'merged',
                'components': [deepcopy(v['bbox']) for v in cluster_views],
                'source_methods': list({v.get('method', 'direct') for v in cluster_views})
            }

            merged_views.append(merged_view)

        return merged_views

    def _finalize_views(self, views, img_width, img_height):
        """
        Post-process views: add normalized coordinates and sort.
        """
        final_views = []
        
        for i, view in enumerate(views):
            x1, y1, x2, y2 = view['bbox']
            
            # Add normalized coordinates
            view['bbox_norm'] = [
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height
            ]
            
            # Add YOLO format (center + width/height)
            cx = (x1 + x2) / 2.0 / img_width
            cy = (y1 + y2) / 2.0 / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height
            
            view['bbox_yolo'] = {
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h
            }
            
            # Add view type based on position
            if y1 < img_height / 3:
                view['position'] = 'top'
            elif y1 < img_height * 2 / 3:
                view['position'] = 'middle'
            else:
                view['position'] = 'bottom'
            
            view['id'] = i
            final_views.append(view)
        
        # Sort by position (top to bottom, left to right)
        final_views.sort(key=lambda v: (v['bbox'][1], v['bbox'][0]))
        
        return final_views
    
    def _visualize_results(self, img, views, save_path, dividers=None, bbox_key='bbox'):
        """
        Draw bounding boxes on image and save result.
        """
        img_vis = img.copy()
        dividers = dividers or []
        
        # Define colors
        base_thickness = max(2, int(img.shape[0] / 1000))
        position_colors = {
            'top': (0, 255, 0),
            'middle': (0, 255, 0),
            'bottom': (0, 255, 0)
        }
        divider_colors = {
            'primary': (0, 0, 255),      # Red for dominant gap
            'secondary': (255, 0, 0)     # Blue for secondary cuts
        }

        for view in views:
            bbox = view.get(bbox_key) or view.get('bbox')
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            position = view.get('position', 'middle')
            color = position_colors.get(position, (0, 0, 255))
            
            # Draw rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, base_thickness)
            
            # Add label
            method = view.get('method', 'direct')
            label = f"View {view['id']} ({method})"
            
            font_scale = max(0.5, min(1.0, img.shape[0] / 2000))
            font_thickness = max(1, int(font_scale * 2))
            
            # Label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(img_vis, (x1, y1 - label_h - 10), 
                         (x1 + label_w + 10, y1), color, -1)
            
            # Label text
            cv2.putText(img_vis, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Draw dividers
        for divider in dividers:
            if isinstance(divider, dict):
                bbox = divider.get('bbox')
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                divider_type = divider.get('type', 'secondary')
                color = divider_colors.get(divider_type, (255, 0, 0))
                label = 'Primary gap' if divider_type == 'primary' else 'Gap'
            else:
                if divider is None:
                    continue
                x1, y1, x2, y2 = divider
                color = divider_colors.get('secondary', (255, 0, 0))
                label = 'Gap'

            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, base_thickness)

            # Optional label for dividers
            label_pos = (x1 + 5, max(20, y1 + 20))
            cv2.putText(
                img_vis,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, min(0.8, img.shape[0] / 3000)),
                color,
                max(1, base_thickness - 1)
            )
        
        # Add summary
        summary = f"Total Views: {len(views)}"
        cv2.putText(img_vis, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Save result if requested
        if save_path:
            cv2.imwrite(save_path, img_vis)
            print(f"Visualization saved to: {save_path}")
        
        return img_vis


# # Example usage
# if __name__ == "__main__":
#     # Create separator instance
#     separator = ViewSeparator(
#         min_area_ratio=0.02,
#         max_area_ratio=0.3,
#         watershed_min_distance=100,
#         edge_margin_ratio=0.03
#     )
    
#     # Process image
#     image_path = "blueprint.jpg"  # Replace with your image path
#     views = separator.separate_views(
#         image_path,
#         visualize=True,
#         save_path='view_separation_result.jpg'
#     )
    
#     # Print results
#     print(f"\nDetected {len(views)} views:")
#     for view in views:
#         print(f"View {view['id']}:")
#         print(f"  Position: {view['position']}")
#         print(f"  Bbox: {view['bbox']}")
#         print(f"  Size: {view['width']}x{view['height']}")
#         print(f"  Area: {view['area']}")
#         print(f"  Method: {view.get('method', 'direct')}")
#         print()