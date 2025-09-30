import cv2
import json
import numpy as np
from collections import defaultdict

class WiredTableDetector:
    DEFAULT_MAX_LONG_SIDE = 3840
    DEFAULT_TEXT_CLASSIFICATION_RULES = {
        'min_text_regions': 4,
        'min_row_groups': 2,
        'min_col_groups': 2,
        'min_sequence_len': 2,
        'min_table_coverage': 0.03,
        'max_table_coverage': 0.8,
        'dense_text_threshold': 0.6,
        'sparse_text_threshold': 0.015,
    }
    def __init__(self):
        """
        Unified table detection system combining fine line detection and table grouping
        """
        self._last_resize_info = None

    def _load_and_resize_image(self, image_path, target_long_side):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        orig_height, orig_width = img.shape[:2]
        max_side = max(orig_width, orig_height)

        if target_long_side <= 0 or max_side <= target_long_side:
            processed = img.copy()
            scale_x = 1.0
            scale_y = 1.0
        else:
            scale = target_long_side / max_side
            new_width = max(1, int(round(orig_width * scale)))
            new_height = max(1, int(round(orig_height * scale)))
            processed = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            scale_x = processed.shape[1] / orig_width
            scale_y = processed.shape[0] / orig_height

        resize_info = {
            'orig_shape': (orig_height, orig_width),
            'processed_shape': processed.shape[:2],
            'scale_x': scale_x,
            'scale_y': scale_y,
            'target_long_side': target_long_side,
        }

        self._last_resize_info = resize_info
        return img, processed, resize_info

    @staticmethod
    def _normalize_bbox(bbox, width, height):
        if width <= 0 or height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = bbox
        return [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]

    @staticmethod
    def _bbox_to_yolo(bbox, width, height):
        if width <= 0 or height <= 0:
            return {'cx': 0.0, 'cy': 0.0, 'w': 0.0, 'h': 0.0}
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return {
            'cx': max(0.0, min(1.0, cx / width)),
            'cy': max(0.0, min(1.0, cy / height)),
            'w': max(0.0, min(1.0, w / width)),
            'h': max(0.0, min(1.0, h / height)),
        }

    def _augment_table_geometry(self, tables, resize_info):
        if not tables:
            return tables

        scale_x = resize_info.get('scale_x', 1.0) or 1.0
        scale_y = resize_info.get('scale_y', 1.0) or 1.0
        orig_height, orig_width = resize_info.get('orig_shape', (1, 1))
        processed_height, processed_width = resize_info.get('processed_shape', (orig_height, orig_width))

        for table in tables:
            bbox_processed = table['bbox']
            x1p, y1p, x2p, y2p = bbox_processed

            width_processed = max(1, x2p - x1p)
            height_processed = max(1, y2p - y1p)
            area_processed = width_processed * height_processed

            table['bbox_processed'] = bbox_processed
            table['width_processed'] = width_processed
            table['height_processed'] = height_processed
            table['area_processed'] = area_processed

            x1 = int(round(x1p / scale_x))
            y1 = int(round(y1p / scale_y))
            x2 = int(round(x2p / scale_x))
            y2 = int(round(y2p / scale_y))

            x1 = max(0, min(orig_width - 1, x1)) if orig_width > 0 else 0
            y1 = max(0, min(orig_height - 1, y1)) if orig_height > 0 else 0
            x2 = max(0, min(orig_width, x2)) if orig_width > 0 else 0
            y2 = max(0, min(orig_height, y2)) if orig_height > 0 else 0

            if x2 <= x1:
                x2 = min(orig_width, x1 + 1) if orig_width > 0 else x1 + 1
            if y2 <= y1:
                y2 = min(orig_height, y1 + 1) if orig_height > 0 else y1 + 1

            width_original = max(1, x2 - x1)
            height_original = max(1, y2 - y1)
            area_original = width_original * height_original

            table['bbox'] = [x1, y1, x2, y2]
            table['width'] = width_original
            table['height'] = height_original
            table['area'] = area_original

            table['bbox_norm'] = self._normalize_bbox(table['bbox'], orig_width, orig_height)
            table['bbox_yolo'] = self._bbox_to_yolo(table['bbox'], orig_width, orig_height)
            table['image_size'] = {
                'original_width': orig_width,
                'original_height': orig_height,
                'processed_width': processed_width,
                'processed_height': processed_height,
            }
            table['scale_factors'] = {
                'scale_x': scale_x,
                'scale_y': scale_y,
            }

        return tables
    
    def _preprocess_for_fine_lines(self, gray_image):
        """
        Enhanced preprocessing specifically for fine line detection
        
        Args:
            gray_image: grayscale input image
            
        Returns:
            preprocessed binary image
        """
        # Method 1: Adaptive threshold for local variations
        adaptive = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 11, 8
        )
        
        # Method 2: Morphological gradient to enhance edges
        kernel = np.ones((2, 2), np.uint8)
        gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        _, gradient_thresh = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
        
        # Method 3: Laplacian edge detection
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, laplacian_thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        
        # Combine all methods
        combined = cv2.bitwise_or(adaptive, gradient_thresh)
        combined = cv2.bitwise_or(combined, laplacian_thresh)
        
        # Clean up noise
        kernel_clean = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_clean)
        
        return combined
    
    def _detect_lines_multi_scale(self, binary_image):
        """
        Detect lines at multiple scales to capture both small and large structures
        
        Args:
            binary_image: preprocessed binary image
            
        Returns:
            combined horizontal and vertical line masks
        """
        height, width = binary_image.shape
        
        # Multi-scale line detection - start with SMALL kernels for small tables
        kernel_sizes_h = [
            max(10, width // 200),   # Very small tables
            max(20, width // 100),   # Small tables  
            max(40, width // 50),    # Medium tables
            max(80, width // 25),    # Large tables
        ]
        
        kernel_sizes_v = [
            max(10, height // 200),  # Very small tables
            max(20, height // 100),  # Small tables
            max(40, height // 50),   # Medium tables  
            max(80, height // 25),   # Large tables
        ]
        
        # Collect all horizontal lines
        all_h_lines = np.zeros_like(binary_image)
        for k_size in kernel_sizes_h:
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
            h_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, h_kernel, iterations=1)
            all_h_lines = cv2.bitwise_or(all_h_lines, h_lines)
        
        # Collect all vertical lines
        all_v_lines = np.zeros_like(binary_image)
        for k_size in kernel_sizes_v:
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
            v_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, v_kernel, iterations=1)
            all_v_lines = cv2.bitwise_or(all_v_lines, v_lines)
        
        return all_h_lines, all_v_lines
    
    def _find_all_rectangles(self, horizontal_lines, vertical_lines, min_area=100):
        """
        Find ALL rectangular regions without aggressive filtering
        
        Args:
            horizontal_lines: horizontal line mask
            vertical_lines: vertical line mask  
            min_area: minimum area threshold (very small)
            
        Returns:
            list of ALL detected rectangles
        """
        # Combine lines with different weights to preserve structure
        combined = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Very gentle morphological operations to preserve small rectangles
        kernel_small = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Find contours with RETR_TREE to get nested structures
        contours_tree, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        
        # Process ALL contours
        for i, contour in enumerate(contours_tree):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Very lenient filtering - accept almost anything rectangular
            if w > 5 and h > 5:  # Minimum size check
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if (w * h) > 0 else 0
                
                # Very broad acceptance criteria
                if (aspect_ratio > 0.05 and aspect_ratio < 50 and 
                    extent > 0.1 and w > 10 and h > 10):
                    
                    rectangles.append({
                        'id': i,
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'width': w,
                        'height': h,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent
                    })
        
        # Sort by area (smallest first to see small tables)
        rectangles.sort(key=lambda x: x['area'])
        
        return rectangles
    
    def _filter_oversized_rectangles(self, rectangles, image_shape, max_area_ratio=0.8):
        """
        Remove rectangles that are too large (likely image boundaries)
        """
        if not rectangles:
            return rectangles
            
        image_area = image_shape[0] * image_shape[1]
        max_allowed_area = image_area * max_area_ratio
        
        filtered = []
        for rect in rectangles:
            if rect['area'] <= max_allowed_area:
                filtered.append(rect)
        
        return filtered
    
    def _filter_edge_rectangles(self, rectangles, image_shape, edge_margin_ratio=0.05):
        """
        Remove rectangles too close to image edges
        """
        if not rectangles:
            return rectangles
            
        height, width = image_shape
        margin_x = width * edge_margin_ratio
        margin_y = height * edge_margin_ratio
        
        filtered = []
        for rect in rectangles:
            x1, y1, x2, y2 = rect['bbox']
            
            # Check if rectangle is too close to any edge
            too_close_to_edge = (
                x1 < margin_x or                    # Too close to left edge
                y1 < margin_y or                    # Too close to top edge
                x2 > width - margin_x or            # Too close to right edge
                y2 > height - margin_y              # Too close to bottom edge
            )
            
            if not too_close_to_edge:
                filtered.append(rect)
        
        return filtered

    def _is_valid_rectangular_contour(self, contour, min_rectangularity=0.85):
        """Check if contour is sufficiently rectangular (avoid triangles/irregular shapes)."""
        if contour is None or len(contour) < 3:
            return False

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area == 0:
            return False

        contour_area = cv2.contourArea(contour)
        extent = contour_area / rect_area

        return extent >= min_rectangularity

    def _validate_rectangle_shape(self, rect, binary_image, min_rectangularity=0.85):
        """Validate if detected rectangle is actually rectangular (not triangular)."""
        x1, y1, x2, y2 = rect['bbox']

        h, w = binary_image.shape
        x1 = max(0, min(w - 1, int(x1))) if w > 0 else 0
        y1 = max(0, min(h - 1, int(y1))) if h > 0 else 0
        x2 = max(0, min(w, int(x2))) if w > 0 else 0
        y2 = max(0, min(h, int(y2))) if h > 0 else 0

        if x2 <= x1 or y2 <= y1:
            return False

        region = binary_image[y1:y2, x1:x2]
        if region.size == 0:
            return False

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        is_rectangular = self._is_valid_rectangular_contour(largest_contour, min_rectangularity)

        if not is_rectangular:
            lx, ly, lw, lh = cv2.boundingRect(largest_contour)
            rect_area = lw * lh
            contour_area = cv2.contourArea(largest_contour)
            extent = contour_area / rect_area if rect_area > 0 else 0

        return is_rectangular

    def _filter_non_rectangular_shapes(self, rectangles, binary_image, min_rectangularity=0.85):
        """Filter out triangular and irregular shapes, keep only rectangular tables."""
        if not rectangles:
            return rectangles

        filtered = []
        filtered_count = 0

        for rect in rectangles:
            if self._validate_rectangle_shape(rect, binary_image, min_rectangularity):
                filtered.append(rect)
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"Filtered {filtered_count} non-rectangular shapes (triangles, etc)")

        return filtered
    
    def _rectangles_adjacent(self, rect1, rect2, tolerance=10):
        """
        Check if two rectangles are adjacent or overlapping (should be merged)
        """
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2
        
        # Check for overlap or adjacency in both dimensions
        x_overlap = not (x2_1 < x1_2 - tolerance or x2_2 < x1_1 - tolerance)
        y_overlap = not (y2_1 < y1_2 - tolerance or y2_2 < y1_1 - tolerance)
        
        # Check horizontal adjacency (left-right touching)
        horizontal_adjacent = (
            (abs(x2_1 - x1_2) <= tolerance or abs(x2_2 - x1_1) <= tolerance) and
            y_overlap  # Vertically overlapping
        )
        
        # Check vertical adjacency (top-bottom touching)
        vertical_adjacent = (
            (abs(y2_1 - y1_2) <= tolerance or abs(y2_2 - y1_1) <= tolerance) and
            x_overlap  # Horizontally overlapping
        )
        
        # Check for significant overlap (should also be merged)
        overlap_adjacent = x_overlap and y_overlap
        
        return horizontal_adjacent or vertical_adjacent or overlap_adjacent
    
    def _merge_adjacent_rectangles_with_size_check(self, rectangles, tolerance=10, image_shape=None, max_area_ratio=0.8):
        """
        Merge adjacent rectangles with size validation - rollback if merged result is too large
        """
        if len(rectangles) <= 1:
            return rectangles
        
        image_area = image_shape[0] * image_shape[1] if image_shape else None
        max_allowed_area = image_area * max_area_ratio if image_area else float('inf')
        
        # Create adjacency graph
        adjacency = defaultdict(set)
        
        for i, rect1 in enumerate(rectangles):
            for j, rect2 in enumerate(rectangles):
                if i != j and self._rectangles_adjacent(rect1['bbox'], rect2['bbox'], tolerance):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        
        # Find connected components using DFS
        visited = set()
        merged_groups = []
        
        def dfs(node, group):
            if node in visited:
                return
            visited.add(node)
            group.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)
        
        for i in range(len(rectangles)):
            if i not in visited:
                group = []
                dfs(i, group)
                if len(group) > 1:  # Only process groups with multiple rectangles
                    merged_groups.append(group)
        
        # Process each group with size validation
        merged_rectangles = []
        used_indices = set()
        
        for group in merged_groups:
            # Calculate potential merged rectangle
            min_x = min(rectangles[i]['bbox'][0] for i in group)
            min_y = min(rectangles[i]['bbox'][1] for i in group)
            max_x = max(rectangles[i]['bbox'][2] for i in group)
            max_y = max(rectangles[i]['bbox'][3] for i in group)
            
            merged_area = (max_x - min_x) * (max_y - min_y)
            
            # Check if merged rectangle would be too large
            if merged_area <= max_allowed_area:
                # Safe to merge
                merged_bbox = [min_x, min_y, max_x, max_y]
                
                # Collect original IDs properly
                merged_from_ids = []
                for i in group:
                    rect = rectangles[i]
                    if 'merged_from' in rect:
                        merged_from_ids.extend(rect['merged_from'])
                    else:
                        merged_from_ids.append(rect['id'])
                
                merged_rect = {
                    'id': f"merged_{len(merged_rectangles)}",
                    'bbox': merged_bbox,
                    'area': merged_area,
                    'width': max_x - min_x,
                    'height': max_y - min_y,
                    'aspect_ratio': (max_x - min_x) / (max_y - min_y) if (max_y - min_y) > 0 else 0,
                    'merged_from': merged_from_ids,
                    'sub_rectangles': len(group),
                    'type': 'merged'
                }
                
                merged_rectangles.append(merged_rect)
                used_indices.update(group)
            # If too large, rollback - don't merge this group, keep individual rectangles
        
        # Add non-merged rectangles (including rolled-back groups)
        for i, rect in enumerate(rectangles):
            if i not in used_indices:
                rect_copy = rect.copy()
                rect_copy['type'] = 'individual'
                rect_copy['sub_rectangles'] = 1
                merged_rectangles.append(rect_copy)
        
        return merged_rectangles
    
    def _rectangle_contains_rectangle(self, outer_rect, inner_rect, margin=5):
        """
        Check if one rectangle completely contains another
        """
        ox1, oy1, ox2, oy2 = outer_rect
        ix1, iy1, ix2, iy2 = inner_rect
        
        return (ox1 <= ix1 + margin and oy1 <= iy1 + margin and 
                ox2 >= ix2 - margin and oy2 >= iy2 - margin)
    
    def _build_hierarchy(self, rectangles):
        """
        Build containment hierarchy of rectangles
        """
        hierarchy = {
            'roots': [],
            'children': defaultdict(list),
            'parents': {}
        }
        
        # Sort rectangles by area (largest first for hierarchy building)
        sorted_rects = sorted(rectangles, key=lambda x: x['area'], reverse=True)
        
        for rect in sorted_rects:
            rect_id = rect['id']
            rect_bbox = rect['bbox']
            
            # Find if this rectangle is contained in any larger rectangle
            parent_found = False
            
            for potential_parent in sorted_rects:
                if (potential_parent['id'] != rect_id and 
                    potential_parent['area'] > rect['area'] and
                    self._rectangle_contains_rectangle(potential_parent['bbox'], rect_bbox)):
                    
                    # Found a parent
                    parent_id = potential_parent['id']
                    hierarchy['children'][parent_id].append(rect_id)
                    hierarchy['parents'][rect_id] = parent_id
                    parent_found = True
                    break
            
            # If no parent found, it's a root
            if not parent_found:
                hierarchy['roots'].append(rect_id)
        
        return hierarchy
    
    def _group_by_containment(self, rectangles):
        """
        Group small rectangles by their containing larger rectangles
        """
        # Build hierarchy
        hierarchy = self._build_hierarchy(rectangles)
        
        # Create rectangle lookup
        rect_lookup = {rect['id']: rect for rect in rectangles}
        
        # Group rectangles by their top-level parents
        groups = {}
        
        for root_id in hierarchy['roots']:
            root_rect = rect_lookup[root_id]
            
            # Collect all descendants
            descendants = []
            
            def collect_descendants(rect_id):
                descendants.append(rect_lookup[rect_id])
                for child_id in hierarchy['children'][rect_id]:
                    collect_descendants(child_id)
            
            collect_descendants(root_id)
            
            groups[root_id] = {
                'main_table': root_rect,
                'sub_rectangles': descendants[1:],  # Exclude the root itself
                'total_sub_rectangles': len(descendants) - 1
            }
        
        return groups
    
    def _create_final_tables(self, grouped_tables):
        """
        Create final table results with classification
        """
        final_tables = []
        
        for group_id, group_data in grouped_tables.items():
            main_table = group_data['main_table']
            sub_count = group_data['total_sub_rectangles']
            
            # Classify table types
            if sub_count == 0:
                table_type = "simple_rectangle"
            elif sub_count < 5:
                table_type = "small_table"
            elif sub_count < 20:
                table_type = "medium_table"
            else:
                table_type = "large_table"
            
            final_table = {
                'id': main_table['id'],
                'bbox': main_table['bbox'],
                'area': main_table['area'],
                'width': main_table['width'],
                'height': main_table['height'],
                'sub_rectangles_count': sub_count,
                'table_type': table_type,
                'merged_from': main_table.get('merged_from', []),
                'confidence': min(1.0, sub_count / 10),  # Simple confidence score
                'content_type': 'unclassified'
            }
            
            final_tables.append(final_table)
        
        # Sort by area (largest first)
        final_tables.sort(key=lambda x: x['area'], reverse=True)
        
        return final_tables
    
    def _is_small_simple_rectangle(self, table, min_size_threshold=5000):
        """
        Check if a table is a small simple rectangle that should be filtered
        
        Args:
            table: table dictionary
            min_size_threshold: minimum area threshold for keeping simple rectangles
            
        Returns:
            bool indicating if table should be filtered out
        """
        is_simple = table['table_type'] == 'simple_rectangle'
        is_small = table['area'] < min_size_threshold
        
        return is_simple and is_small
    
    def _filter_small_simple_rectangles(self, final_tables, min_size_threshold=5000, image_area=None):
        """
        Filter out small simple rectangles (gray colored ones)
        
        Args:
            final_tables: list of detected tables
            min_size_threshold: minimum area threshold for keeping simple rectangles:
                              - int/float > 1: absolute pixel area (e.g., 5000)
                              - float <= 1: percentage of image area (e.g., 0.01 = 1%)
            image_area: total image area for percentage calculation
            
        Returns:
            filtered list of tables
        """
        # Calculate actual threshold based on input type
        if min_size_threshold <= 1.0:
            # Percentage mode
            if image_area is None:
                raise ValueError("image_area required for percentage-based min_size_threshold")
            actual_threshold = image_area * min_size_threshold
            print(f"Using {min_size_threshold:.3%} of image area as threshold: ")
        else:
            # Absolute pixel mode
            actual_threshold = min_size_threshold
            if image_area:
                percentage = actual_threshold / image_area * 100
                print(f"Using absolute threshold: {actual_threshold:.0f} pixels ({percentage:.1f}% of image)")
        
        filtered_tables = []
        filtered_count = 0
        
        for table in final_tables:
            if not self._is_small_simple_rectangle(table, actual_threshold):
                filtered_tables.append(table)
            else:
                filtered_count += 1
        
        if filtered_count > 0:
            print(f"Filtered out <{filtered_count}> small simple rectangles")
        
        return filtered_tables

    def _merge_nearby_small_regions(self, tables, image_area=None, area_ratio_threshold=0.05, distance_threshold_ratio=0.02):
        """Merge nearby small tables into a minimal rectangle when close to each other."""
        if not tables:
            return tables

        if image_area is None:
            image_area = 0

        area_threshold = image_area * area_ratio_threshold if image_area else None

        large_tables = []
        small_tables = []

        for table in tables:
            area = table.get('area', 0)
            if area_threshold is not None and area < area_threshold:
                small_tables.append(table)
            else:
                large_tables.append(table)

        if not small_tables:
            return tables

        groups = []
        visited = set()

        def bbox_distance(bbox1, bbox2):
            x11, y11, x12, y12 = bbox1
            x21, y21, x22, y22 = bbox2

            dx = max(0, max(x21 - x12, x11 - x22))
            dy = max(0, max(y21 - y12, y11 - y22))
            return (dx ** 2 + dy ** 2) ** 0.5

        # Estimate scale for distance threshold (using image diagonal)
        if image_area and tables:
            sample_table = tables[0]
            size_info = sample_table.get('image_size') or {}
            width = size_info.get('original_width')
            height = size_info.get('original_height')
            if width and height:
                diag = (width ** 2 + height ** 2) ** 0.5
                distance_threshold = diag * distance_threshold_ratio
            else:
                distance_threshold = 50
        else:
            distance_threshold = 50

        for idx, table in enumerate(small_tables):
            if idx in visited:
                continue
            group = [table]
            visited.add(idx)

            for jdx, other in enumerate(small_tables[idx + 1:], start=idx + 1):
                if jdx in visited:
                    continue
                dist = bbox_distance(table['bbox'], other['bbox'])
                if dist <= distance_threshold:
                    group.append(other)
                    visited.add(jdx)

            groups.append(group)

        merged_tables = []

        for group in groups:
            if len(group) == 1:
                merged_tables.append(group[0])
                continue

            x1 = min(t['bbox'][0] for t in group)
            y1 = min(t['bbox'][1] for t in group)
            x2 = max(t['bbox'][2] for t in group)
            y2 = max(t['bbox'][3] for t in group)

            merged_table = group[0].copy()
            merged_table['bbox'] = [x1, y1, x2, y2]
            merged_table['width'] = max(1, x2 - x1)
            merged_table['height'] = max(1, y2 - y1)
            merged_table['area'] = merged_table['width'] * merged_table['height']
            merged_table['sub_rectangles_count'] = sum(t.get('sub_rectangles_count', 1) for t in group)
            merged_table['merged_small_regions'] = [t.get('id') for t in group]

            size_info = merged_table.get('image_size')
            if size_info:
                orig_w = size_info.get('original_width')
                orig_h = size_info.get('original_height')
                if orig_w and orig_h:
                    merged_table['bbox_norm'] = self._normalize_bbox(merged_table['bbox'], orig_w, orig_h)
                    merged_table['bbox_yolo'] = self._bbox_to_yolo(merged_table['bbox'], orig_w, orig_h)

            merged_tables.append(merged_table)

        # Merge final result: large tables + merged small groups
        return large_tables + merged_tables
    
    def _visualize_results(self, img, final_tables, save_path='table_detection_result.jpg'):
        """
        Visualize the final table detection results
        """
        img_vis = img.copy()
        
        # Color coding by table type
        type_colors = {
            'simple_rectangle': (0, 0, 255),  # Red
            'small_table': (0, 255, 0),          # Green
            'medium_table': (0, 165, 255),       # Orange
            'large_table': (255, 0, 0),          # Blue
        }
        
        for table in final_tables:
            bbox = table['bbox']
            x1, y1, x2, y2 = bbox
            
            table_type = table.get('table_type', 'simple_rectangle')
            color = type_colors.get(table_type, (255, 255, 255))
            
            # Draw rectangle with thickness based on importance
            thickness = max(2, int(img.shape[0] / 1000))
            if table['sub_rectangles_count'] > 10:
                thickness *= 2
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            label = f"{table['table_type'][:6]}({table['sub_rectangles_count']})"
            font_scale = max(0.5, min(1.5, img.shape[0] / 3000))
            
            # Label background for visibility
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img_vis, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.putText(img_vis, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        cv2.imwrite(save_path, img_vis)
        return img_vis

    def _show_all_rectangles_debug(self, image_path, *,
                                   min_area=50,
                                   small_area_ratio=0.02,
                                   target_long_side=None,
                                   save_path='all_rectangles_debug.jpg',
                                   draw_ids=True,
                                   max_rectangles_visualize=1500):
        """Debug helper: show and export all initially detected rectangles and highlight "small rectangles".

        Parameters:
            image_path: input image path
            min_area: minimum area passed to _find_all_rectangles
            small_area_ratio: threshold to consider a rectangle "small" (fraction of processed image area)
            target_long_side: processing long side (defaults to class default)
            save_path: output debug image path
            draw_ids: whether to draw rectangle IDs
            max_rectangles_visualize: avoid drawing too many rectangles in extreme cases

        Returns:
            dict containing:
                'rectangles': list of all rectangles
                'small_rectangles': list of small rectangles
                'save_path': saved file path
                'resize_info': resize information
                'small_area_threshold': area threshold (pixels)
        """
        import cv2
        import numpy as np

        if target_long_side is None:
            target_long_side = self.DEFAULT_MAX_LONG_SIDE

        # 1. Load and resize
        original_img, processed_img, resize_info = self._load_and_resize_image(image_path, target_long_side)
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        ph, pw = processed_img.shape[:2]
        processed_area = ph * pw
        small_area_threshold = processed_area * small_area_ratio

        # 2. Preprocess, detect lines, and get initial rectangles
        binary = self._preprocess_for_fine_lines(gray)
        h_lines, v_lines = self._detect_lines_multi_scale(binary)
        rectangles = self._find_all_rectangles(h_lines, v_lines, min_area)

        if not rectangles:
            print("[debug] No initial rectangles detected")
            return {
                'rectangles': [],
                'small_rectangles': [],
                'save_path': None,
                'resize_info': resize_info,
                'small_area_threshold': small_area_threshold,
            }

        # 3. Mark small rectangles
        small_rectangles = []
        for r in rectangles:
            r_area = r.get('area')
            if r_area is None:
                x1, y1, x2, y2 = r['bbox']
                r_area = (x2 - x1) * (y2 - y1)
                r['area'] = r_area
            if r_area < small_area_threshold:
                r['__is_small'] = True
                small_rectangles.append(r)
            else:
                r['__is_small'] = False

        # 4. Visualization
        vis = processed_img.copy()

        # Limit visualization count (sort by area, show smallest first)
        to_draw = sorted(rectangles, key=lambda x: x['area'])
        if len(to_draw) > max_rectangles_visualize:
            print(f"[debug] Too many rectangles ({len(to_draw)}), truncating to first {max_rectangles_visualize} for visualization")
            to_draw = to_draw[:max_rectangles_visualize]

        for rect in to_draw:
            x1, y1, x2, y2 = rect['bbox']
            is_small = rect['__is_small']
            color = (0, 255, 255) if is_small else (160, 160, 160)
            thickness = 2 if is_small else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            if draw_ids:
                rid = rect.get('id')
                if rid is not None:
                    label = f"{rid}:{int(rect['area'])}"
                    cv2.putText(vis, label[:18], (x1+2, max(12, y1+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        legend_lines = [
            f"Total rects: {len(rectangles)}",
            f"Small rects(<{small_area_ratio*100:.2f}% area): {len(small_rectangles)}",
            f"Processed size: {pw}x{ph}",
        ]
        y0 = 20
        for line in legend_lines:
            cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            y0 += 24

        cv2.imwrite(save_path, vis)
        print(f"[debug] All initial rectangles debug image saved: {save_path}")

        return {
            'rectangles': rectangles,
            'small_rectangles': small_rectangles,
            'save_path': save_path,
            'resize_info': resize_info,
            'small_area_threshold': small_area_threshold,
        }

    def detect_tables(self, image_path, merge_tolerance=15, max_area_ratio=0.8, max_merge_passes=2,
                     min_area=50, save_visualization=True, filter_edge_rectangles=True,
                     edge_margin_ratio=0.01, path='table_detection_result.jpg', 
                     filter_small_simple=True, min_size_threshold=5000,
                     filter_non_rectangular=True, min_rectangularity=0.85,
                     merge_small_regions=True, small_region_area_ratio=0.05,
                     small_region_distance_ratio=0.02,
                     target_long_side=None):
        """
        Main public interface - detect all tables in image
        
        Args:
            image_path: path to input image
            merge_tolerance: tolerance for merging adjacent rectangles (default: 15)
            max_area_ratio: max area ratio to filter oversized rectangles (default: 0.8 = 80%)
            min_area: minimum rectangle area to consider (default: 50)
            save_visualization: whether to save result visualization (default: True)
            filter_edge_rectangles: whether to filter rectangles near edges (default: True)
            edge_margin_ratio: margin ratio from edges for filtering (default: 0.01 = 1%)
            path: save path for visualization result
            filter_small_simple: whether to filter small simple rectangles (default: True)
            min_size_threshold: minimum area threshold for keeping simple rectangles:
                              - int/float > 1: absolute pixel area (e.g., 5000)
                              - float <= 1: percentage of image area (e.g., 0.01 = 1%)
            filter_non_rectangular: whether to reject triangular/irregular contours before merging.
            min_rectangularity: minimum contour extent to accept a rectangle (default: 0.85).
            merge_small_regions: whether to merge nearby small tables into minimal bounding boxes.
            small_region_area_ratio: area ratio threshold (of image) that defines "small" tables.
            small_region_distance_ratio: proximity threshold (relative to image diagonal) for merging.
            target_long_side: max dimension (long side) for processing resolution; defaults to 4K (3840).
            
        Returns:
            list of detected tables with metadata
        """
        if target_long_side is None:
            target_long_side = self.DEFAULT_MAX_LONG_SIDE

        original_img, processed_img, resize_info = self._load_and_resize_image(image_path, target_long_side)
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        proc_height, proc_width = processed_img.shape[:2]
        image_area = proc_height * proc_width
        print(
            f"Processing image (original {original_img.shape[1]}x{original_img.shape[0]} -> "
            f"processed {proc_width}x{proc_height})"
        )
        # Step 1: Preprocess for fine line detection
        binary = self._preprocess_for_fine_lines(gray)
        
        # Step 2: Multi-scale line detection
        h_lines, v_lines = self._detect_lines_multi_scale(binary)
        
        # Step 3: Find all rectangles
        rectangles = self._find_all_rectangles(h_lines, v_lines, min_area)
        print(f"Detected {len(rectangles)} initial rectangles")
        if not rectangles:
            return []
        
        # Step 4a: Filter oversized rectangles (BEFORE merging)
        filtered_rectangles = self._filter_oversized_rectangles(
            rectangles, processed_img.shape[:2], max_area_ratio
        )
        
        # Step 4b: Filter edge rectangles if enabled
        if filter_edge_rectangles:
            filtered_rectangles = self._filter_edge_rectangles(
                filtered_rectangles, processed_img.shape[:2], edge_margin_ratio
            )

        if filter_non_rectangular:
            filtered_rectangles = self._filter_non_rectangular_shapes(
                filtered_rectangles, binary, min_rectangularity
            )
        
        if not filtered_rectangles:
            return []
        
        # Step 5: Smart merge with size validation - run multiple passes if needed
        merged_rectangles = filtered_rectangles
        # Limit merge passes to avoid infinite loops
        
        for pass_num in range(max_merge_passes):
            before_count = len(merged_rectangles)
            
            # Increase tolerance slightly with each pass to catch missed adjacencies
            current_tolerance = merge_tolerance * (1 + pass_num * 0.5)
            current_max_area = max_area_ratio + (0.1 * pass_num)
            
            merged_rectangles = self._merge_adjacent_rectangles_with_size_check(
                merged_rectangles, current_tolerance, processed_img.shape[:2], current_max_area
            )
            
            after_count = len(merged_rectangles)
            print(f"Merge pass {pass_num + 1}: {before_count} -> {after_count} rectangles")
            
            # Stop if no more merging occurred
            if after_count >= before_count:
                break
        
        if not merged_rectangles:
            return []
        
        # Step 6: Build hierarchy and group by containment
        grouped_tables = self._group_by_containment(merged_rectangles)
        
        # Step 7: Create final table results
        final_tables = self._create_final_tables(grouped_tables)
        
        # Step 8: Filter small simple rectangles if enabled
        if filter_small_simple:
            final_tables = self._filter_small_simple_rectangles(final_tables, min_size_threshold, image_area)

        if merge_small_regions:
            final_tables = self._merge_nearby_small_regions(
                final_tables,
                image_area=image_area,
                area_ratio_threshold=small_region_area_ratio,
                distance_threshold_ratio=small_region_distance_ratio
            )

        # Step 9: Normalize geometry back to original resolution and percentages
        final_tables = self._augment_table_geometry(final_tables, resize_info)

        # Step 10: Visualize results on original resolution
        if save_visualization:
            self._visualize_results(original_img, final_tables, path)

        return final_tables

    def export_tables(self, tables, save_path, include_sub_rectangles=False):
        """
        Export detected table metadata to a JSON file.

        Args:
            tables: list returned by detect_tables
            save_path: output filename (JSON)
            include_sub_rectangles: when True, include merged_from provenance

        Returns:
            The path written for convenience
        """
        payload = []

        for table in tables:
            record = {
                'id': table['id'],
                'bbox': table['bbox'],
                'bbox_norm': [round(v, 6) for v in table.get('bbox_norm', [])] if table.get('bbox_norm') else None,
                'bbox_yolo': {k: round(v, 6) for k, v in table.get('bbox_yolo', {}).items()} if table.get('bbox_yolo') else None,
                'bbox_processed': table.get('bbox_processed'),
                'width': table['width'],
                'height': table['height'],
                'width_processed': table.get('width_processed'),
                'height_processed': table.get('height_processed'),
                'area': table['area'],
                'table_type': table.get('table_type'),
                'sub_rectangles_count': table.get('sub_rectangles_count', 0),
                'confidence': table.get('confidence', 0.0),
                'content_type': table.get('content_type'),
                'image_size': table.get('image_size'),
                'scale_factors': table.get('scale_factors'),
            }

            if record['bbox_norm'] is None:
                record.pop('bbox_norm')
            if record['bbox_yolo'] is None:
                record.pop('bbox_yolo')

            if include_sub_rectangles:
                record['merged_from'] = table.get('merged_from', [])

            summary = table.get('text_summary')
            if summary:
                record['text_overlap_ratio'] = round(summary.get('text_coverage', 0.0), 4)
                record['alignment_score'] = summary.get('alignment_score')
                record['text_count'] = summary.get('text_count')

            payload.append(record)

        with open(save_path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        return save_path

    def _compute_frame_text_summary(self, table, frame_info, config):
        bbox = table.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        frame_area = float(width * height)

        text_regions = [r for r in frame_info.get('text_regions', [])
                        if isinstance(r, dict) and r.get('bbox')]

        text_area = 0.0
        text_widths = []
        text_heights = []

        for region in text_regions:
            rx1, ry1, rx2, ry2 = region['bbox']
            rw = max(0, rx2 - rx1)
            rh = max(0, ry2 - ry1)
            text_area += rw * rh
            text_widths.append(rw)
            text_heights.append(rh)

        coverage = text_area / frame_area if frame_area > 0 else 0.0

        alignment = frame_info.get('alignment') or {}
        rows = alignment.get('rows') or []
        columns = alignment.get('columns') or []

        row_count = frame_info.get('row_count', len(rows))
        column_count = frame_info.get('column_count', len(columns))
        max_row_len = max((len(row) for row in rows), default=0)
        max_col_len = max((len(col) for col in columns), default=0)

        row_sequences = sum(1 for row in rows if len(row) >= config['min_sequence_len'])
        column_sequences = sum(1 for col in columns if len(col) >= config['min_sequence_len'])

        text_count = frame_info.get('text_count', len(text_regions))

        has_horizontal = max_row_len >= config['min_sequence_len']
        has_vertical = max_col_len >= config['min_sequence_len']

        dense_text = coverage >= config['dense_text_threshold']
        sparse_text = (coverage <= config['sparse_text_threshold'] or
                       text_count < config['min_text_regions'])

        alignment_ok = (row_count >= config['min_row_groups'] and
                        column_count >= config['min_col_groups'])
        sequence_ok = has_horizontal and has_vertical
        coverage_ok = (config['min_table_coverage'] <= coverage <= config['max_table_coverage'])

        if alignment_ok and sequence_ok and coverage_ok:
            classification = 'table'
        elif sparse_text:
            classification = 'frame_outline'
        elif dense_text and not alignment_ok:
            classification = 'text_block'
        else:
            classification = 'ambiguous_frame'

        summary = {
            'text_count': text_count,
            'row_count': row_count,
            'column_count': column_count,
            'row_sequence_count': row_sequences,
            'column_sequence_count': column_sequences,
            'max_row_length': max_row_len,
            'max_column_length': max_col_len,
            'text_area': float(text_area),
            'frame_area': frame_area,
            'text_coverage': coverage,
            'alignment_score': min(row_count, column_count),
            'has_horizontal_sequence': has_horizontal,
            'has_vertical_sequence': has_vertical,
            'dense_text': dense_text,
            'sparse_text': sparse_text,
            'classification': classification,
        }

        if text_widths:
            summary['avg_text_width'] = float(np.mean(text_widths))
            summary['avg_text_height'] = float(np.mean(text_heights))
        else:
            summary['avg_text_width'] = 0.0
            summary['avg_text_height'] = 0.0

        summary['text_density'] = (text_count / frame_area) if frame_area > 0 else 0.0

        return summary

    def _attach_text_metrics_to_tables(self, tables, frame_analysis, config):
        frame_lookup = {}

        for info in frame_analysis:
            frame_id = info.get('frame_id')
            if frame_id is None:
                frame = info.get('frame')
                if isinstance(frame, dict):
                    frame_id = frame.get('id')
            if frame_id is not None:
                frame_lookup[frame_id] = info

        for table in tables:
            table_id = table.get('id')
            frame_info = frame_lookup.get(table_id)

            if frame_info is None:
                bbox = table.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                frame_area = float(max(1, x2 - x1) * max(1, y2 - y1))
                summary = {
                    'text_count': 0,
                    'row_count': 0,
                    'column_count': 0,
                    'row_sequence_count': 0,
                    'column_sequence_count': 0,
                    'max_row_length': 0,
                    'max_column_length': 0,
                    'text_area': 0.0,
                    'frame_area': frame_area,
                    'text_coverage': 0.0,
                    'alignment_score': 0,
                    'has_horizontal_sequence': False,
                    'has_vertical_sequence': False,
                    'dense_text': False,
                    'sparse_text': True,
                    'classification': 'unmatched'
                }
                summary['avg_text_width'] = 0.0
                summary['avg_text_height'] = 0.0
                summary['text_density'] = 0.0
                table['text_summary'] = summary
                table['content_type'] = table.get('content_type', 'unclassified')
                continue

            summary = self._compute_frame_text_summary(table, frame_info, config)
            table['text_summary'] = summary
            table['content_type'] = summary['classification']
            frame_info['text_summary'] = summary
            frame_info['content_type'] = summary['classification']

        return tables

    def detect_tables_with_text(self, image_path, text_detector, *,
                                text_method='all', min_quality_score=0.3,
                                alignment_tolerance=20, assignment_method='center',
                                overlap_threshold=0.1, min_regions=2,
                                return_unassigned=True, text_kwargs=None,
                                classification_rules=None, filter_non_tables=False,
                                **table_kwargs):
        """Detect wired tables and analyze text within each frame to reduce text-level complexity.

        Args:
            image_path: input image path.
            text_detector: instance of ``TextPositionTableDetector`` or compatible API.
            text_method: text detection backend passed to ``extract_text_regions``.
            min_quality_score: filter threshold for text regions.
            alignment_tolerance: tolerance for per-frame alignment analysis.
            assignment_method: 'center' or 'overlap' (see ``assign_text_regions_to_frames``).
            overlap_threshold: overlap ratio when using 'overlap' assignment.
            min_regions: minimum region count required to run alignment inside a frame.
            return_unassigned: whether to include text regions that didn't match any frame.
            text_kwargs: optional dict forwarded to ``analyze_text_within_frames`` (e.g., custom params).
            classification_rules: optional overrides for text-based table classification thresholds.
            filter_non_tables: when True, only keep candidates classified as real tables.
            **table_kwargs: forwarded to ``detect_tables`` (e.g., merge_tolerance, save_visualization).

        Returns:
            dict with keys:
                'tables': wired table detection output
                'text_regions': list of all text regions detected
                'frame_text': per-frame analysis structure from ``analyze_text_within_frames``
        """
        target_long_side = table_kwargs.get('target_long_side', None)
        tables = self.detect_tables(image_path, **table_kwargs)

        if not tables:
            return {
                'tables': [],
                'text_regions': [],
                'frame_text': {'frames': [], 'unassigned_regions': []},
                'meta': {'resize_info': self._last_resize_info}
            }

        text_regions = text_detector.extract_text_regions(
            image_path,
            min_quality_score=min_quality_score,
            method=text_method,
            target_long_side=target_long_side or self.DEFAULT_MAX_LONG_SIDE,
            resize_info=self._last_resize_info
        )

        analysis_kwargs = {
            'alignment_tolerance': alignment_tolerance,
            'min_regions': min_regions,
            'method': assignment_method,
            'overlap_threshold': overlap_threshold,
        }
        if text_kwargs:
            analysis_kwargs.update(text_kwargs)
        if self._last_resize_info:
            analysis_kwargs['scale_factors'] = self._last_resize_info

        analysis = text_detector.analyze_text_within_frames(
            tables,
            text_regions,
            **analysis_kwargs
        )

        if not return_unassigned and 'unassigned_regions' in analysis:
            analysis = analysis.copy()
            analysis['unassigned_regions'] = []

        classification_config = self.DEFAULT_TEXT_CLASSIFICATION_RULES.copy()
        if classification_rules:
            classification_config.update(classification_rules)

        tables = self._attach_text_metrics_to_tables(tables, analysis['frames'], classification_config)

        if filter_non_tables:
            tables = [table for table in tables if table.get('content_type') == 'table']
            kept_ids = {table.get('id') for table in tables}
            if kept_ids:
                filtered_frames = []
                for frame_info in analysis.get('frames', []):
                    frame_id = frame_info.get('frame_id')
                    if frame_id in kept_ids:
                        filtered_frames.append(frame_info)
                        continue
                    frame = frame_info.get('frame')
                    if isinstance(frame, dict) and frame.get('id') in kept_ids:
                        filtered_frames.append(frame_info)
                analysis = analysis.copy()
                analysis['frames'] = filtered_frames

        return {
            'tables': tables,
            'text_regions': text_regions,
            'frame_text': analysis,
            'meta': {'resize_info': self._last_resize_info}
        }
    
    