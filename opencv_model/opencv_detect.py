import cv2
import numpy as np
from collections import defaultdict

class SimpleTableDetector:
    def __init__(self):
        """
        Simple table detector using OpenCV
        """
        pass
    
    def detect_tables_opencv(self, image_path, min_table_area=5000):
        """
        Detect tables using OpenCV morphological operations
        
        Args:
            image_path: path to input image
            min_table_area: minimum area for table detection
            
        Returns:
            list of table bounding boxes
        """
        print(f"Loading image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Define kernels for detecting horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Dilate to connect nearby elements
        table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and extract table regions
        table_boxes = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_table_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and size
                aspect_ratio = w / h
                if aspect_ratio > 0.3 and w > 80 and h > 40:
                    table_boxes.append({
                        'id': i,
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'width': w,
                        'height': h,
                        'aspect_ratio': aspect_ratio
                    })
        
        print(f"Found {len(table_boxes)} potential tables")
        return table_boxes, img
    
    def visualize_detection(self, img, table_boxes, save_path=None):
        """
        Draw bounding boxes on image to visualize detection results
        
        Args:
            img: input image array
            table_boxes: list of detected table boxes
            save_path: optional path to save visualization
            
        Returns:
            annotated image
        """
        img_vis = img.copy()
        
        for table in table_boxes:
            bbox = table['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Table {table['id']} ({table['width']}x{table['height']})"
            cv2.putText(img_vis, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
            print(f"Visualization saved: {save_path}")
        
        return img_vis


class FineLineTableDetector:
    def __init__(self):
        """
        Enhanced table detector focused on detecting ALL small rectangles first
        """
        pass
    
    def preprocess_for_fine_lines(self, gray_image):
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
    
    def detect_lines_multi_scale(self, binary_image):
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
        
        print(f"Using horizontal kernel sizes: {kernel_sizes_h}")
        print(f"Using vertical kernel sizes: {kernel_sizes_v}")
        
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
    
    def find_all_rectangles(self, horizontal_lines, vertical_lines, min_area=100):
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
        
        # Find contours with different retrieval modes
        contours_external, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours_external)} external contours")
        print(f"Found {len(contours_tree)} total contours")
        
        rectangles = []
        
        # Process ALL contours (not just external)
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
                        'extent': extent,
                        'hierarchy_level': hierarchy[0][i] if hierarchy is not None else None
                    })
        
        # Sort by area (smallest first to see small tables)
        rectangles.sort(key=lambda x: x['area'])
        
        print(f"Rectangle areas range: {rectangles[0]['area']:.0f} to {rectangles[-1]['area']:.0f}")
        
        return rectangles
    
    def detect_all_rectangles(self, image_path, min_area=100, debug=False):
        """
        Main method to detect ALL rectangles without aggressive filtering
        
        Args:
            image_path: path to input image
            min_area: minimum area (very small)
            debug: save debug images
            
        Returns:
            all detected rectangles and original image
        """
        print(f"Loading image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Enhanced preprocessing
        binary = self.preprocess_for_fine_lines(gray)
        if debug:
            cv2.imwrite('debug_01_binary.jpg', binary)
            print("Saved debug_01_binary.jpg")
        
        # Step 2: Multi-scale line detection
        h_lines, v_lines = self.detect_lines_multi_scale(binary)
        if debug:
            cv2.imwrite('debug_02_horizontal.jpg', h_lines)
            cv2.imwrite('debug_03_vertical.jpg', v_lines)
            print("Saved debug line detection images")
        
        # Step 3: Find ALL rectangles
        rectangles = self.find_all_rectangles(h_lines, v_lines, min_area)
        
        print(f"Total rectangles found: {len(rectangles)}")
        
        return rectangles, img
    
    def visualize_all_detections(self, img, rectangles, max_display=50, save_path=None):
        """
        Visualize ALL detections (or top N)
        
        Args:
            img: original image
            rectangles: all detected rectangles
            max_display: maximum number to display
            save_path: save path
            
        Returns:
            annotated image
        """
        img_vis = img.copy()
        
        # Display largest rectangles first
        display_rects = rectangles[-max_display:] if len(rectangles) > max_display else rectangles
        
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Light Green
            (255, 128, 0),  # Orange
        ]
        
        for i, rect in enumerate(display_rects):
            bbox = rect['bbox']
            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, max(1, int(img.shape[0]/2000)))
            
            # Add small label (only for reasonable sized rectangles)
            if rect['width'] > 50 and rect['height'] > 20:
                label = f"{i+1}"
                font_scale = max(0.3, min(1.0, img.shape[0]/5000))
                cv2.putText(img_vis, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
            print(f"Visualization saved: {save_path}")
        
        return img_vis

# # Test function to find ALL rectangles
# def test_all_rectangle_detection(image_path, debug=True):
#     """
#     Test detection of ALL rectangles
#     """
#     detector = FineLineTableDetector()
    
#     try:
#         # Detect ALL rectangles
#         rectangles, img = detector.detect_all_rectangles(image_path, min_area=50, debug=debug)
        
#         # Show statistics
#         print(f"\n=== Detection Statistics ===")
#         print(f"Total rectangles: {len(rectangles)}")
        
#         if rectangles:
#             areas = [r['area'] for r in rectangles]
#             widths = [r['width'] for r in rectangles]  
#             heights = [r['height'] for r in rectangles]
            
#             print(f"Area range: {min(areas):.0f} - {max(areas):.0f}")
#             print(f"Width range: {min(widths)} - {max(widths)}")
#             print(f"Height range: {min(heights)} - {max(heights)}")
            
#             # Show different size categories
#             small = [r for r in rectangles if r['area'] < 1000]
#             medium = [r for r in rectangles if 1000 <= r['area'] < 10000] 
#             large = [r for r in rectangles if r['area'] >= 10000]
            
#             print(f"Small rectangles (<1000 area): {len(small)}")
#             print(f"Medium rectangles (1K-10K area): {len(medium)}")
#             print(f"Large rectangles (>10K area): {len(large)}")
            
#             # Show some examples from each category
#             if small:
#                 print(f"Small example: {small[0]['width']}x{small[0]['height']}")
#             if medium:
#                 print(f"Medium example: {medium[0]['width']}x{medium[0]['height']}")  
#             if large:
#                 print(f"Large example: {large[0]['width']}x{large[0]['height']}")
        
#         # Visualize results
#         img_vis = detector.visualize_all_detections(img, rectangles, max_display=100, 
#                                                    save_path='all_rectangles_result.jpg')
        
#         return rectangles, img
        
#     except Exception as e:
#         print(f"Error during detection: {e}")
#         import traceback
#         traceback.print_exc()
#         return [], None

import cv2
import numpy as np
from collections import defaultdict

class TableMultiplier:
    def __init__(self):
        """
        Complete table detection system with simplified interface
        """
        pass
    
    def _preprocess_for_fine_lines(self, gray_image):
        """
        Enhanced preprocessing specifically for fine line detection
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
        removed_count = 0
        
        for rect in rectangles:
            if rect['area'] <= max_allowed_area:
                filtered.append(rect)
            else:
                area_ratio = rect['area'] / image_area
                print(f"Removing oversized rectangle: {rect['width']}x{rect['height']} "
                      f"(area ratio: {area_ratio:.2%})")
                removed_count += 1
        
        if removed_count > 0:
            print(f"Filtered out {removed_count} oversized rectangles (>{max_area_ratio:.0%} of image)")
        
        return filtered
    
    def _filter_edge_rectangles(self, rectangles, image_shape, edge_margin_ratio=0.05):
        """
        Remove rectangles too close to image edges
        
        Args:
            rectangles: list of rectangles
            image_shape: (height, width) of image
            edge_margin_ratio: margin ratio from edges (0.05 = 5% from each edge)
        """
        if not rectangles:
            return rectangles
            
        height, width = image_shape
        margin_x = width * edge_margin_ratio
        margin_y = height * edge_margin_ratio
        
        filtered = []
        removed_count = 0
        
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
            else:
                print(f"Removing edge rectangle: {rect['width']}x{rect['height']} at ({x1}, {y1})")
                removed_count += 1
        
        if removed_count > 0:
            print(f"Filtered out {removed_count} edge rectangles (within {edge_margin_ratio:.1%} of edges)")
        
        return filtered
    
    def _rectangles_adjacent(self, rect1, rect2, tolerance=10):
        """
        Check if two rectangles are adjacent (sharing borders)
        """
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2
        
        # Check horizontal adjacency (left-right touching)
        horizontal_adjacent = (
            (abs(x2_1 - x1_2) <= tolerance or abs(x2_2 - x1_1) <= tolerance) and
            not (y2_1 < y1_2 - tolerance or y2_2 < y1_1 - tolerance)  # Vertically overlapping
        )
        
        # Check vertical adjacency (top-bottom touching)
        vertical_adjacent = (
            (abs(y2_1 - y1_2) <= tolerance or abs(y2_2 - y1_1) <= tolerance) and
            not (x2_1 < x1_2 - tolerance or x2_2 < x1_1 - tolerance)  # Horizontally overlapping
        )
        
        return horizontal_adjacent or vertical_adjacent
    
    def _merge_adjacent_rectangles_with_size_check(self, rectangles, tolerance=10, image_shape=None, max_area_ratio=0.8):
        """
        Merge adjacent rectangles with size validation - rollback if merged result is too large
        
        Args:
            rectangles: list of rectangle dictionaries
            tolerance: pixel tolerance for adjacency
            image_shape: (height, width) for size validation
            max_area_ratio: maximum allowed area ratio for merged rectangles
            
        Returns:
            list of merged rectangles (with rollback for oversized merges)
        """
        if len(rectangles) <= 1:
            return rectangles
        
        image_area = image_shape[0] * image_shape[1] if image_shape else None
        max_allowed_area = image_area * max_area_ratio if image_area else float('inf')
        
        print(f"Starting merge with size check (max area ratio: {max_area_ratio:.1%})")
        
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
        
        print(f"Found {len(merged_groups)} groups of adjacent rectangles")
        
        # Process each group with size validation
        merged_rectangles = []
        used_indices = set()
        rollback_count = 0
        
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
                
                merged_rect = {
                    'id': f"merged_{len(merged_rectangles)}",
                    'bbox': merged_bbox,
                    'area': merged_area,
                    'width': max_x - min_x,
                    'height': max_y - min_y,
                    'aspect_ratio': (max_x - min_x) / (max_y - min_y) if (max_y - min_y) > 0 else 0,
                    'merged_from': [rectangles[i]['id'] for i in group],
                    'sub_rectangles': len(group),
                    'type': 'merged'
                }
                
                merged_rectangles.append(merged_rect)
                used_indices.update(group)
                
                print(f"Merged {len(group)} rectangles into {max_x - min_x}x{max_y - min_y} "
                      f"(area ratio: {merged_area/image_area:.2%})")
            else:
                # Rollback - don't merge this group, keep individual rectangles
                area_ratio = merged_area / image_area if image_area else 0
                print(f"ROLLBACK: Merge of {len(group)} rectangles would create oversized result "
                      f"{max_x - min_x}x{max_y - min_y} (area ratio: {area_ratio:.2%})")
                rollback_count += 1
                # Note: don't add these indices to used_indices, so they'll be kept as individuals
        
        # Add non-merged rectangles (including rolled-back groups)
        for i, rect in enumerate(rectangles):
            if i not in used_indices:
                rect_copy = rect.copy()
                rect_copy['type'] = 'individual'
                rect_copy['sub_rectangles'] = 1
                merged_rectangles.append(rect_copy)
        
        print(f"Merge complete: {len(merged_rectangles)} rectangles "
              f"({rollback_count} groups rolled back due to size)")
        
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
    
    def _visualize_results(self, img, final_tables, save_path=None):
        """
        Visualize the final table detection results
        """
        img_vis = img.copy()
        
        # Color coding by table type
        type_colors = {
            'simple_rectangle': (128, 128, 128),  # Gray
            'small_table': (0, 255, 0),          # Green
            'medium_table': (0, 165, 255),       # Orange
            'large_table': (0, 0, 255),          # Red
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
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
        
        return img_vis
    
    def detect_tables(self, image_path, merge_tolerance=15, max_area_ratio=0.8, 
                     min_area=50, save_visualization=True, filter_edge_rectangles=True,
                     edge_margin_ratio=0.05):
        """
        Main public interface - detect all tables in image
        
        Args:
            image_path: path to input image
            merge_tolerance: tolerance for merging adjacent rectangles (default: 15)
            max_area_ratio: max area ratio to filter oversized rectangles (default: 0.8 = 80%)
            min_area: minimum rectangle area to consider (default: 50)
            save_visualization: whether to save result visualization (default: True)
            filter_edge_rectangles: whether to filter rectangles near edges (default: True)
            edge_margin_ratio: margin ratio from edges for filtering (default: 0.05 = 5%)
            
        Returns:
            list of detected tables with metadata
        """
        print(f"Processing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Preprocess for fine line detection
        binary = self._preprocess_for_fine_lines(gray)
        
        # Step 2: Multi-scale line detection
        h_lines, v_lines = self._detect_lines_multi_scale(binary)
        
        # Step 3: Find all rectangles
        rectangles = self._find_all_rectangles(h_lines, v_lines, min_area)
        print(f"Initial rectangles detected: {len(rectangles)}")
        
        if not rectangles:
            print("No rectangles detected!")
            return []
        
        # Step 4a: Filter oversized rectangles (BEFORE merging)
        filtered_rectangles = self._filter_oversized_rectangles(
            rectangles, img.shape[:2], max_area_ratio
        )
        print(f"After size filtering: {len(filtered_rectangles)} rectangles")
        
        # Step 4b: Filter edge rectangles if enabled
        if filter_edge_rectangles:
            filtered_rectangles = self._filter_edge_rectangles(
                filtered_rectangles, img.shape[:2], edge_margin_ratio
            )
            print(f"After edge filtering: {len(filtered_rectangles)} rectangles")
        
        if not filtered_rectangles:
            print("No valid rectangles after filtering!")
            return []
        
        # Step 5: Smart merge with size validation (no rollback needed now!)
        merged_rectangles = self._merge_adjacent_rectangles_with_size_check(
            filtered_rectangles, merge_tolerance, img.shape[:2], max_area_ratio
        )
        print(f"After smart merging: {len(merged_rectangles)} rectangles")
        
        if not merged_rectangles:
            print("No valid rectangles after merging!")
            return []
        
        # Step 6: Build hierarchy and group by containment
        grouped_tables = self._group_by_containment(merged_rectangles)
        
        # Step 7: Create final table results
        final_tables = self._create_final_tables(grouped_tables)
        
        # Step 8: Visualize results
        if save_visualization:
            self._visualize_results(img, final_tables, 'table_detection_result.jpg')
            print("Result saved: table_detection_result.jpg")
        
        # Print summary with area percentages
        print(f"\n=== Detection Complete ===")
        print(f"Found {len(final_tables)} tables:")
        for i, table in enumerate(final_tables, 1):
            area_ratio = table['area'] / (img.shape[0] * img.shape[1]) * 100
            print(f"  {i}. {table['table_type']}: {table['width']}x{table['height']} pixels, "
                  f"{table['sub_rectangles_count']} sub-rectangles ({area_ratio:.1f}%)")
        
        return final_tables