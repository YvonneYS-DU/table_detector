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



class TextPositionTableDetector_old:
    def __init__(self):
        """
        Table detector for borderless tables based on text position analysis
        """
        pass
    
    def _extract_text_regions(self, image_path, min_quality_score=0.3):
        """
        Extract text regions prioritizing contour method with quality-based filtering
        
        Args:
            image_path: path to input image
            min_quality_score: minimum confidence threshold for filtering
            
        Returns:
            list of text regions with bbox and basic info
        """
        print(f"Extracting text regions from: {image_path}")
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use enhanced contour method
        print("Using enhanced contour method for detection...")
        text_regions = self._extract_text_contours_enhanced(gray)
        
        print(f"Detected: {len(text_regions)} regions before filtering")
        
        # Apply quality filtering using passed parameter
        filtered_regions = self._filter_regions_by_quality(text_regions, min_quality_score)
        
        print(f"Final result: {len(filtered_regions)} text regions after quality filtering")
        return filtered_regions
    
    def debug_text_detection(self, image_path, save_debug_images=True):
        """
        Debug function optimized for contour-based detection
        
        Args:
            image_path: path to input image
            save_debug_images: whether to save intermediate debug images
            
        Returns:
            detailed debug information
        """
        print(f"=== DEBUG: Enhanced Contour-based Text Detection ===")
        print(f"Processing: {image_path}")
        
        # Step 1: Focus on contour method
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get enhanced contour results
        contour_regions = self._extract_text_contours_enhanced(gray)
        
        print(f"\nPrimary Detection Results:")
        print(f"  Enhanced contour method: {len(contour_regions)} regions")
        
        # Quality analysis
        high_quality = [r for r in contour_regions if r.get('quality_score', 0) >= 0.7]
        medium_quality = [r for r in contour_regions if 0.4 <= r.get('quality_score', 0) < 0.7]
        low_quality = [r for r in contour_regions if r.get('quality_score', 0) < 0.4]
        
        print(f"  Quality breakdown: {len(high_quality)} high, {len(medium_quality)} medium, {len(low_quality)} low")
        
        # Apply quality filtering
        filtered_regions = self._filter_regions_by_quality(contour_regions, min_quality_score=0.3)
        
        # Deduplication
        final_regions = self._deduplicate_text_regions_fast(filtered_regions)
        
        if save_debug_images:
            # Save quality-coded debug images
            self._save_quality_debug(img, contour_regions, 'debug_contour_quality.jpg')
            self._save_method_debug(img, final_regions, 'debug_final_regions.jpg', 'Final Filtered')
        
        # Step 2: Analyze alignment
        alignment_info = self._analyze_text_alignment_fast(final_regions, alignment_tolerance=20)
        
        print(f"\nAlignment Analysis:")
        print(f"  Potential table rows: {len(alignment_info['rows'])}")
        print(f"  Potential table columns: {len(alignment_info['columns'])}")
        
        # Step 3: Detect tables
        tables = self._detect_grid_patterns(
            alignment_info['rows'], 
            alignment_info['columns'],
            min_intersections=4
        )
        
        print(f"\nTable Detection Results:")
        print(f"  Found {len(tables)} borderless tables")
        
        for i, table in enumerate(tables):
            bbox = table['bbox']
            print(f"    Table {i+1}: {table['rows']}x{table['columns']}, "
                  f"confidence: {table['confidence']:.3f}")
        
        # Step 4: Create final visualization
        if save_debug_images and tables:
            self._visualize_text_based_tables(image_path, tables, final_regions)
        
        return {
            'text_regions': final_regions,
            'alignment_info': alignment_info,
            'tables': tables,
            'method_breakdown': {
                'contour_enhanced': len(contour_regions),
                'after_quality_filter': len(filtered_regions),
                'final_deduplicated': len(final_regions)
            }
        }
    
    def _save_quality_debug(self, img, regions, filename):
        """
        Save debug image with quality-based color coding
        """
        debug_img = img.copy()
        
        for region in regions:
            x1, y1, x2, y2 = [int(coord) for coord in region['bbox']]
            quality = region.get('quality_score', 0.5)
            
            # Color by quality: red=low, yellow=medium, green=high
            if quality >= 0.7:
                color = (0, 255, 0)  # High quality - Green
            elif quality >= 0.4:
                color = (0, 255, 255)  # Medium quality - Yellow
            else:
                color = (0, 0, 255)  # Low quality - Red
            
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 1)
            
            # Add quality score
            cv2.putText(debug_img, f"{quality:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add legend
        cv2.putText(debug_img, f"Quality Debug: {len(regions)} regions", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, "Green=High, Yellow=Med, Red=Low", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(filename, debug_img)
        print(f"Quality debug image saved: {filename}")
        
        return debug_img
    
    def _extract_text_mser(self, gray_image):
        """
        Extract text regions using MSER
        """
        # Create MSER detector with correct parameter names
        mser = cv2.MSER_create()
        
        # Set parameters using setter methods
        mser.setMinArea(50)      # Minimum area of text regions
        mser.setMaxArea(10000)   # Maximum area of text regions
        mser.setDelta(5)
        
        regions, _ = mser.detectRegions(gray_image)
        text_regions = []
        
        for i, region in enumerate(regions):
            if len(region) > 10:  # Filter very small regions
                x_coords = region[:, 0]
                y_coords = region[:, 1]
                
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                width, height = x2 - x1, y2 - y1
                
                # Filter by text-like characteristics
                if self._is_text_like_region(width, height):
                    text_regions.append({
                        'id': f"mser_{i}",
                        'bbox': [x1, y1, x2, y2],
                        'text': f"region_{i}",
                        'confidence': 0.7,
                        'center': [(x1 + x2)/2, (y1 + y2)/2],
                        'method': 'mser'
                    })
        
        return text_regions
    
    def _extract_text_morphological(self, gray_image):
        """
        Extract text regions using morphological operations
        """
        # Threshold image
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            if self._is_text_like_region(w, h):
                text_regions.append({
                    'id': f"morph_{i}",
                    'bbox': [x, y, x + w, y + h],
                    'text': f"text_{i}",
                    'confidence': 0.6,
                    'center': [(x + x + w)/2, (y + y + h)/2],
                    'method': 'morphological'
                })
        
        return text_regions
    
    def _extract_text_contours(self, gray_image):
        """
        Extract text regions using contour analysis
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby components
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by text characteristics
            if self._is_text_like_region(w, h, area):
                text_regions.append({
                    'id': f"contour_{i}",
                    'bbox': [x, y, x + w, y + h],
                    'text': f"area_{i}",
                    'confidence': 0.5,
                    'center': [(x + x + w)/2, (y + y + h)/2],
                    'method': 'contour'
                })
        
        return text_regions
    
    def _is_text_like_region(self, width, height, area=None):
        """
        Check if a region has text-like characteristics
        
        Args:
            width: region width
            height: region height
            area: region area (optional)
            
        Returns:
            True if region is text-like
        """
        if width < 10 or height < 8:  # Too small
            return False
        
        if width > 1000 or height > 200:  # Too large
            return False
        
        aspect_ratio = width / height
        
        # Text usually has horizontal aspect ratio
        if aspect_ratio < 0.3 or aspect_ratio > 20:
            return False
        
        # Check area consistency if provided
        if area is not None:
            expected_area = width * height
            if area < expected_area * 0.2:  # Too sparse
                return False
        
        return True
    
    def _deduplicate_text_regions_fast(self, all_regions, overlap_threshold=0.7):
        """
        Fast deduplication using spatial hashing to avoid O(n²) complexity
        
        Args:
            all_regions: list of all detected regions
            overlap_threshold: overlap ratio threshold for deduplication
            
        Returns:
            deduplicated list of text regions
        """
        if not all_regions:
            return []
        
        print(f"Fast deduplication of {len(all_regions)} regions...")
        
        # Sort by confidence (highest first)
        sorted_regions = sorted(all_regions, key=lambda r: r['confidence'], reverse=True)
        
        # Use spatial hashing for faster collision detection
        grid_size = 50  # pixels per grid cell
        spatial_hash = defaultdict(list)
        
        deduplicated = []
        
        for region in sorted_regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Calculate grid cells this region occupies
            grid_x1, grid_y1 = int(x1 // grid_size), int(y1 // grid_size)
            grid_x2, grid_y2 = int(x2 // grid_size), int(y2 // grid_size)
            
            # Check only nearby regions in adjacent grid cells
            is_duplicate = False
            for gx in range(grid_x1 - 1, grid_x2 + 2):
                for gy in range(grid_y1 - 1, grid_y2 + 2):
                    for existing in spatial_hash[(gx, gy)]:
                        overlap_ratio = self._calculate_overlap_ratio(region['bbox'], existing['bbox'])
                        if overlap_ratio > overlap_threshold:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break
            
            if not is_duplicate:
                deduplicated.append(region)
                # Add to spatial hash
                for gx in range(grid_x1, grid_x2 + 1):
                    for gy in range(grid_y1, grid_y2 + 1):
                        spatial_hash[(gx, gy)].append(region)
        
        print(f"Fast deduplication complete: {len(deduplicated)} unique regions")
        return deduplicated
    
    def _filter_regions_by_quality(self, regions, min_quality_score=0.3):
        """
        Soft filter regions based on confidence only
        
        Args:
            regions: list of detected regions
            min_quality_score: minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            quality-filtered regions
        """
        print(f"Quality filtering {len(regions)} regions with confidence threshold {min_quality_score}...")
        
        # Use only confidence as quality score
        for region in regions:
            region['quality_score'] = region.get('confidence', 0.5)
        
        # Filter by confidence threshold (soft filtering)
        quality_filtered = [r for r in regions if r['quality_score'] >= min_quality_score]
        
        print(f"Quality filtering complete: kept {len(quality_filtered)} regions "
              f"(removed {len(regions) - len(quality_filtered)})")
        
        return quality_filtered
    
    def _extract_text_contours_enhanced(self, gray_image):
        """
        Enhanced contour-based text detection with better filtering
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # Multiple edge detection approaches
        # Method 1: Standard Canny
        edges1 = cv2.Canny(blurred, 50, 150)
        
        # Method 2: Adaptive Canny
        median_val = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        edges2 = cv2.Canny(blurred, lower, upper)
        
        # Combine edge detection results
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Connect nearby text components with different kernel sizes
        kernels = [
            np.ones((2, 6), np.uint8),  # Horizontal connection
            np.ones((3, 3), np.uint8),  # General connection
        ]
        
        all_contours = []
        for kernel in kernels:
            dilated = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        text_regions = []
        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Enhanced text region validation
            if self._is_text_like_region_enhanced(w, h, area, contour):
                # Calculate more accurate confidence based on contour properties
                confidence = self._calculate_contour_confidence(contour, area, w, h)
                
                text_regions.append({
                    'id': f"contour_{i}",
                    'bbox': [x, y, x + w, y + h],
                    'text': f"area_{i}",
                    'confidence': confidence,
                    'center': [(x + x + w)/2, (y + y + h)/2],
                    'method': 'contour_enhanced',
                    'extent': area / (w * h) if (w * h) > 0 else 0
                })
        
        return text_regions
    
    def _is_text_like_region_enhanced(self, width, height, area, contour):
        """
        Enhanced text region validation with contour analysis
        """
        if width < 8 or height < 6:  # Too small
            return False
        
        if width > 2000 or height > 500:  # Too large
            return False
        
        aspect_ratio = width / height if height > 0 else 0
        
        # More flexible aspect ratio for different text orientations
        if aspect_ratio < 0.2 or aspect_ratio > 50:
            return False
        
        # Area consistency check
        bbox_area = width * height
        if area < bbox_area * 0.1:  # Too sparse
            return False
        
        # Contour complexity check (text should have reasonable complexity)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            if compactness > 50:  # Too complex/jagged
                return False
        
        return True
    
    def _calculate_contour_confidence(self, contour, area, width, height):
        """
        Calculate confidence score based on contour properties
        """
        base_confidence = 0.6
        
        # Size bonus
        if 100 <= area <= 10000:
            base_confidence += 0.2
        elif 50 <= area <= 20000:
            base_confidence += 0.1
        
        # Aspect ratio bonus
        aspect_ratio = width / height if height > 0 else 0
        if 1.5 <= aspect_ratio <= 10:
            base_confidence += 0.2
        elif 0.8 <= aspect_ratio <= 20:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _analyze_text_alignment_fast(self, text_regions, alignment_tolerance=20):
        """
        Fast text alignment analysis using sorted arrays instead of nested loops
        
        Args:
            text_regions: list of text regions
            alignment_tolerance: pixel tolerance for alignment detection
            
        Returns:
            dict with row and column information
        """
        if not text_regions:
            return {'rows': [], 'columns': []}
        
        print(f"Fast alignment analysis of {len(text_regions)} regions...")
        
        # Sort once and reuse for both row and column grouping
        y_sorted = sorted(text_regions, key=lambda r: r['center'][1])
        x_sorted = sorted(text_regions, key=lambda r: r['center'][0])
        
        # Fast grouping using sorted order
        rows = self._group_sorted_regions(y_sorted, 1, alignment_tolerance)  # Group by Y
        columns = self._group_sorted_regions(x_sorted, 0, alignment_tolerance)  # Group by X
        
        # Filter groups with minimum elements
        rows = [row for row in rows if len(row) >= 2]
        columns = [col for col in columns if len(col) >= 2]
        
        print(f"Fast alignment found {len(rows)} rows and {len(columns)} columns")
        return {'rows': rows, 'columns': columns}
    
    def _group_sorted_regions(self, sorted_regions, coord_index, tolerance):
        """
        Group already-sorted regions by coordinate with single pass
        
        Args:
            sorted_regions: regions sorted by coordinate
            coord_index: 0 for X, 1 for Y
            tolerance: grouping tolerance
            
        Returns:
            list of groups
        """
        if not sorted_regions:
            return []
        
        groups = []
        current_group = [sorted_regions[0]]
        current_coord = sorted_regions[0]['center'][coord_index]
        
        for region in sorted_regions[1:]:
            region_coord = region['center'][coord_index]
            
            if abs(region_coord - current_coord) <= tolerance:
                current_group.append(region)
            else:
                groups.append(current_group)
                current_group = [region]
                current_coord = region_coord
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _group_by_coordinate(self, text_regions, coordinate, tolerance):
        """
        Group text regions by x or y coordinate alignment
        
        Args:
            text_regions: list of text regions
            coordinate: 'x' or 'y' for grouping direction
            tolerance: pixel tolerance for grouping
            
        Returns:
            list of groups (each group is a list of text regions)
        """
        coord_index = 0 if coordinate == 'x' else 1
        
        # Sort by coordinate
        sorted_regions = sorted(text_regions, key=lambda r: r['center'][coord_index])
        
        groups = []
        current_group = [sorted_regions[0]]
        current_coord = sorted_regions[0]['center'][coord_index]
        
        for region in sorted_regions[1:]:
            region_coord = region['center'][coord_index]
            
            if abs(region_coord - current_coord) <= tolerance:
                # Same alignment group
                current_group.append(region)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [region]
                current_coord = region_coord
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _detect_grid_patterns(self, rows, columns, min_intersections=4):
        """
        Detect rectangular grid patterns from row and column alignments
        
        Args:
            rows: list of row groups
            columns: list of column groups
            min_intersections: minimum intersections to consider a valid table
            
        Returns:
            list of detected table grids
        """
        tables = []
        
        print(f"Detecting grid patterns from {len(rows)} rows and {len(columns)} columns")
        
        # For each combination of rows, check for grid pattern
        for row_indices in self._get_consecutive_groups(rows, min_size=2):
            selected_rows = [rows[i] for i in row_indices]
            
            # Find columns that intersect with these rows
            intersecting_cols = []
            for col_idx, column in enumerate(columns):
                intersections = 0
                for row in selected_rows:
                    if self._has_intersection(row, column):
                        intersections += 1
                
                if intersections >= len(selected_rows) * 0.7:  # At least 70% intersection
                    intersecting_cols.append(col_idx)
            
            if len(intersecting_cols) >= 2:  # At least 2 columns
                # Calculate table boundary
                all_text_regions = []
                for row in selected_rows:
                    all_text_regions.extend(row)
                
                if len(all_text_regions) >= min_intersections:
                    table_bbox = self._calculate_bounding_box(all_text_regions)
                    
                    tables.append({
                        'bbox': table_bbox,
                        'rows': len(selected_rows),
                        'columns': len(intersecting_cols),
                        'text_regions': all_text_regions,
                        'confidence': self._calculate_grid_confidence(selected_rows, intersecting_cols)
                    })
        
        # Sort by confidence and remove overlapping tables
        tables.sort(key=lambda t: t['confidence'], reverse=True)
        filtered_tables = self._remove_overlapping_tables(tables)
        
        print(f"Detected {len(filtered_tables)} grid-based tables")
        return filtered_tables
    
    def _get_consecutive_groups(self, items, min_size=2):
        """
        Get consecutive groups of items for table detection
        """
        groups = []
        for start in range(len(items)):
            for end in range(start + min_size, len(items) + 1):
                groups.append(list(range(start, end)))
        return groups
    
    def _has_intersection(self, row, column):
        """
        Check if a row and column have intersecting text regions
        """
        row_x_ranges = [(r['bbox'][0], r['bbox'][2]) for r in row]
        col_y_ranges = [(r['bbox'][1], r['bbox'][3]) for r in column]
        
        # Check for any intersection
        for row_x1, row_x2 in row_x_ranges:
            for col_y1, col_y2 in col_y_ranges:
                # Find if any column region intersects with row region
                for col_region in column:
                    col_x1, col_y1, col_x2, col_y2 = col_region['bbox']
                    if not (col_x2 < row_x1 or col_x1 > row_x2):  # X overlap
                        return True
        return False
    
    def _calculate_bounding_box(self, text_regions):
        """
        Calculate bounding box for a list of text regions
        """
        if not text_regions:
            return [0, 0, 0, 0]
        
        x1 = min(r['bbox'][0] for r in text_regions)
        y1 = min(r['bbox'][1] for r in text_regions)
        x2 = max(r['bbox'][2] for r in text_regions)
        y2 = max(r['bbox'][3] for r in text_regions)
        
        return [x1, y1, x2, y2]
    
    def _calculate_grid_confidence(self, rows, columns):
        """
        Calculate confidence score for detected grid
        """
        # Base score
        score = 0.5
        
        # More rows/columns = higher confidence
        score += min(len(rows) * 0.1, 0.3)
        score += min(len(columns) * 0.1, 0.3)
        
        # Regular spacing bonus
        if len(rows) > 2:
            row_y_coords = [r[0]['center'][1] for r in rows]
            row_spacing_var = np.var(np.diff(sorted(row_y_coords)))
            if row_spacing_var < 100:  # Low variance in spacing
                score += 0.2
        
        return min(score, 1.0)
    
    def _remove_overlapping_tables(self, tables, overlap_threshold=0.5):
        """
        Remove overlapping table detections
        """
        filtered = []
        
        for table in tables:
            bbox = table['bbox']
            is_duplicate = False
            
            for existing in filtered:
                existing_bbox = existing['bbox']
                overlap_ratio = self._calculate_overlap_ratio(bbox, existing_bbox)
                
                if overlap_ratio > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(table)
        
        return filtered
    
    def _calculate_overlap_ratio(self, bbox1, bbox2):
        """
        Calculate overlap ratio between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _visualize_text_based_tables(self, image_path, tables, text_regions=None, save_path='text_tables.jpg'):
        """
        Visualize detected text-based tables with detailed annotations
        """
        img = cv2.imread(image_path)
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        # Draw all detected text regions first (in light gray)
        if text_regions:
            for text_region in text_regions:
                tx1, ty1, tx2, ty2 = [int(coord) for coord in text_region['bbox']]
                cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (128, 128, 128), 1)
                
                # Add small method label
                method = text_region.get('method', 'unknown')
                cv2.putText(img, method[:4], (tx1, ty1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
        
        # Draw detected tables (in bright colors)
        for i, table in enumerate(tables):
            bbox = table['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = colors[i % len(colors)]
            
            # Draw table boundary (thick)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            
            # Draw individual text regions in this table (thinner, same color)
            for text_region in table['text_regions']:
                tx1, ty1, tx2, ty2 = [int(coord) for coord in text_region['bbox']]
                cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, 2)
            
            # Add comprehensive label
            label = f"Table{i+1}: {table['rows']}x{table['columns']} (conf:{table['confidence']:.2f})"
            
            # Label background for visibility
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.putText(img, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add summary text
        summary = f"Total: {len(text_regions) if text_regions else 0} text regions, {len(tables)} tables"
        cv2.putText(img, summary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, summary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, img)
        print(f"Annotated visualization saved: {save_path}")
        
        return img
    
    def debug_text_detection(self, image_path, save_debug_images=True):
        """
        Debug function to show all intermediate results
        
        Args:
            image_path: path to input image
            save_debug_images: whether to save intermediate debug images
            
        Returns:
            detailed debug information
        """
        print(f"=== DEBUG: Text-based Table Detection ===")
        print(f"Processing: {image_path}")
        
        # Step 1: Extract text regions with method breakdown
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get results from each method separately
        mser_regions = self._extract_text_mser(gray)
        morph_regions = self._extract_text_morphological(gray)
        contour_regions = self._extract_text_contours(gray)
        
        print(f"\nText Region Detection Results:")
        print(f"  MSER method: {len(mser_regions)} regions")
        print(f"  Morphological method: {len(morph_regions)} regions")
        print(f"  Contour method: {len(contour_regions)} regions")
        
        # Combine and deduplicate
        all_regions = mser_regions + morph_regions + contour_regions
        # text_regions = self._deduplicate_text_regions_fast(all_regions)
        print(f"  After deduplication: {len(all_regions)} regions")
        
        if save_debug_images:
            # Save individual method results
            self._save_method_debug(img, mser_regions, 'debug_mser_regions.jpg', 'MSER')
            self._save_method_debug(img, morph_regions, 'debug_morph_regions.jpg', 'Morphological')
            self._save_method_debug(img, contour_regions, 'debug_contour_regions.jpg', 'Contour')
            self._save_method_debug(img, all_regions, 'debug_all_text_regions.jpg', 'All Combined')

        # Step 2: Analyze alignment
        alignment_info = self._analyze_text_alignment_fast(all_regions, alignment_tolerance=20)

        print(f"\nAlignment Analysis:")
        print(f"  Potential table rows: {len(alignment_info['rows'])}")
        print(f"  Potential table columns: {len(alignment_info['columns'])}")
        
        # Show row and column details
        for i, row in enumerate(alignment_info['rows']):
            y_coord = int(row[0]['center'][1])
            print(f"    Row {i+1}: {len(row)} text regions at y≈{y_coord}")
        
        for i, col in enumerate(alignment_info['columns']):
            x_coord = int(col[0]['center'][0])
            print(f"    Column {i+1}: {len(col)} text regions at x≈{x_coord}")
        
        # Step 3: Detect tables
        tables = self._detect_grid_patterns(
            alignment_info['rows'], 
            alignment_info['columns'],
            min_intersections=4
        )
        
        print(f"\nTable Detection Results:")
        print(f"  Found {len(tables)} borderless tables")
        
        for i, table in enumerate(tables):
            bbox = table['bbox']
            print(f"    Table {i+1}:")
            print(f"      Size: {table['rows']}x{table['columns']}")
            print(f"      Bbox: ({int(bbox[0])}, {int(bbox[1])}) to ({int(bbox[2])}, {int(bbox[3])})")
            print(f"      Confidence: {table['confidence']:.3f}")
            print(f"      Text regions: {len(table['text_regions'])}")
        
        # Step 4: Create final visualization
        if save_debug_images:
            final_img = self._visualize_text_based_tables(image_path, tables, all_regions)
        
        return {
            'text_regions': all_regions,
            'alignment_info': alignment_info,
            'tables': tables,
            'method_breakdown': {
                'mser': len(mser_regions),
                'morphological': len(morph_regions),
                'contour': len(contour_regions),
                'final': len(all_regions)
            }
        }
    
    def _save_method_debug(self, img, regions, filename, method_name):
        """
        Save debug image for a specific detection method
        """
        debug_img = img.copy()
        
        # Color code by method
        method_colors = {
            'MSER': (0, 255, 0),
            'Morphological': (255, 0, 0), 
            'Contour': (0, 0, 255),
            'All Combined': (255, 255, 0)
        }
        
        color = method_colors.get(method_name, (128, 128, 128))
        
        for region in regions:
            x1, y1, x2, y2 = [int(coord) for coord in region['bbox']]
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            
            # Add confidence or method info
            conf = region.get('confidence', 0.5)
            cv2.putText(debug_img, f"{conf:.1f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title
        cv2.putText(debug_img, f"{method_name}: {len(regions)} regions", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imwrite(filename, debug_img)
        print(f"Debug image saved: {filename}")
        
        return debug_img
    
    def detect_borderless_tables(self, image_path, alignment_tolerance=20, min_intersections=4, 
                                save_visualization=True):
        """
        Main function to detect borderless tables based on text positions
        
        Args:
            image_path: path to input image
            alignment_tolerance: pixel tolerance for text alignment
            min_intersections: minimum text intersections for valid table
            save_visualization: whether to save result visualization
            
        Returns:
            list of detected borderless tables
        """
        print(f"Detecting borderless tables in: {image_path}")
        
        # Step 1: Extract text regions
        text_regions = self._extract_text_regions(image_path)
        
        if not text_regions:
            print("No text regions found!")
            return []
        
        # Step 2: Analyze text alignment
        alignment_info = self._analyze_text_alignment_fast(text_regions, alignment_tolerance)
        
        # Step 3: Detect grid patterns
        tables = self._detect_grid_patterns(
            alignment_info['rows'], 
            alignment_info['columns'],
            min_intersections
        )
        
        # Step 4: Visualize results
        if save_visualization and tables:
            self._visualize_text_based_tables(image_path, tables)
        
        # Print summary
        print(f"\n=== Borderless Table Detection Complete ===")
        print(f"Found {len(tables)} borderless tables:")
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table['rows']}x{table['columns']} table "
                  f"(confidence: {table['confidence']:.2f})")
        
        return tables

