import cv2
import json
import numpy as np
from collections import defaultdict

class WiredTableDetector:
    def __init__(self):
        """
        Unified table detection system combining fine line detection and table grouping
        """
        pass
    
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
                'confidence': min(1.0, sub_count / 10)  # Simple confidence score
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
            min_size_threshold: minimum area threshold - can be:
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
            print(f"Using {min_size_threshold:.1%} of image area as threshold: {actual_threshold:.0f} pixels")
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
            print(f"Filtered out {filtered_count} small simple rectangles")
        
        return filtered_tables
    
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
    
    def detect_tables(self, image_path, merge_tolerance=15, max_area_ratio=0.8, 
                     min_area=50, save_visualization=True, filter_edge_rectangles=True,
                     edge_margin_ratio=0.01, path='table_detection_result.jpg', 
                     filter_small_simple=True, min_size_threshold=5000):
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
            
        Returns:
            list of detected tables with metadata
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_area = img.shape[0] * img.shape[1]  # Calculate total image area
        print(f"Processing image of size: ({image_area} pixels)")
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
            rectangles, img.shape[:2], max_area_ratio
        )
        
        # Step 4b: Filter edge rectangles if enabled
        if filter_edge_rectangles:
            filtered_rectangles = self._filter_edge_rectangles(
                filtered_rectangles, img.shape[:2], edge_margin_ratio
            )
        
        if not filtered_rectangles:
            return []
        
        # Step 5: Smart merge with size validation - run multiple passes if needed
        merged_rectangles = filtered_rectangles
        max_merge_passes = 3  # Limit merge passes to avoid infinite loops
        
        for pass_num in range(max_merge_passes):
            before_count = len(merged_rectangles)
            
            # Increase tolerance slightly with each pass to catch missed adjacencies
            current_tolerance = merge_tolerance * (1 + pass_num * 0.5)
            current_max_area = max_area_ratio + (0.1 * pass_num)
            
            merged_rectangles = self._merge_adjacent_rectangles_with_size_check(
                merged_rectangles, current_tolerance, img.shape[:2], current_max_area
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
        
        # Step 9: Visualize results
        if save_visualization:
            self._visualize_results(img, final_tables, path)
        
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
                'width': table['width'],
                'height': table['height'],
                'area': table['area'],
                'table_type': table.get('table_type'),
                'sub_rectangles_count': table.get('sub_rectangles_count', 0),
                'confidence': table.get('confidence', 0.0)
            }

            if include_sub_rectangles:
                record['merged_from'] = table.get('merged_from', [])

            payload.append(record)

        with open(save_path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        return save_path

    def detect_tables_with_text(self, image_path, text_detector, *,
                                text_method='all', min_quality_score=0.3,
                                alignment_tolerance=20, assignment_method='center',
                                overlap_threshold=0.1, min_regions=2,
                                return_unassigned=True, text_kwargs=None, **table_kwargs):
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
            **table_kwargs: forwarded to ``detect_tables`` (e.g., merge_tolerance, save_visualization).

        Returns:
            dict with keys:
                'tables': wired table detection output
                'text_regions': list of all text regions detected
                'frame_text': per-frame analysis structure from ``analyze_text_within_frames``
        """
        tables = self.detect_tables(image_path, **table_kwargs)

        if not tables:
            return {
                'tables': [],
                'text_regions': [],
                'frame_text': {'frames': [], 'unassigned_regions': []}
            }

        text_regions = text_detector.extract_text_regions(
            image_path, min_quality_score=min_quality_score, method=text_method
        )

        analysis_kwargs = {
            'alignment_tolerance': alignment_tolerance,
            'min_regions': min_regions,
            'method': assignment_method,
            'overlap_threshold': overlap_threshold,
        }
        if text_kwargs:
            analysis_kwargs.update(text_kwargs)

        analysis = text_detector.analyze_text_within_frames(
            tables,
            text_regions,
            **analysis_kwargs
        )

        if not return_unassigned and 'unassigned_regions' in analysis:
            analysis = analysis.copy()
            analysis['unassigned_regions'] = []

        return {
            'tables': tables,
            'text_regions': text_regions,
            'frame_text': analysis
        }