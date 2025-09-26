import cv2
import numpy as np
"""text_detect_open_cv
=================================================
字符 / 文本区域检测与（无框）表格定位辅助模块

ZH 简介
--------
本模块聚焦于“基于文本位置”来辅助检测无边框表格（borderless tables）。核心思想：
1. 使用多种文本候选区域提取方法（MSER / 形态学 / 轮廓 / 增强轮廓）。
2. 统一输出标准化的文本区域结构（bbox / center / confidence / method 等）。
3. 对文本区域做质量过滤 + 去重 + 行列对齐分析。
4. 通过行列交叉模式推断潜在的无框表格候选区域。

EN Overview
-----------
This module provides text (character/word-like) region extraction utilities to support
borderless table detection based purely on spatial alignment of textual elements.
It offers multiple detection backends (MSER, Morphological, Contour, Enhanced Contour)
and a lightweight pipeline to: detect -> filter -> deduplicate -> alignment analysis ->
grid (table) hypothesis generation.

TextPositionTableDetector
    目标：
        - detect_borderless_tables(image_path, ...) (尚未完成) 未来返回表格候选。
        - debug_text_detection(...) / debug_text_detection (两个名字，后者是更完整调试接口，建议保留一个)。
        - assign_text_regions_to_frames(frames, text_regions, ...) 将文本候选分配到指定 frame 内。
        - analyze_text_within_frames(frames, text_regions, ...) 在每个 frame 内部独立做行列分析以降低整体复杂度。
    关键内部阶段：
        a) _extract_text_* 系列: 不同方法提取文本候选
        b) _filter_regions_by_quality: 置信度过滤
        c) _deduplicate_text_regions_fast: 去重
        d) _analyze_text_alignment_fast: 行列对齐分组
        e) _detect_grid_patterns: 根据行列交叉推测表格

Data Structures / 数据结构
---------------------------
Text Region (dict):
    {
        'id': str,                # 唯一标识
        'bbox': [x1,y1,x2,y2],     # 左上与右下坐标 (int)
        'center': [cx,cy],         # 中心 (float)
        'confidence': float,       # 0~1 的置信度
        'quality_score': float,    # （过滤阶段写入）
        'method': str,             # 生成方法: mser / morph / contour / contour_enhanced 等
        'text': str | None,        # 可选（OCR 后补充）
    }

Alignment Info (dict):
    {
        'rows': List[List[text_region]],
        'columns': List[List[text_region]]
    }

Table Candidate (dict) (未来在 _detect_grid_patterns 中生成):
    {
        'id': str,
        'bbox': [x1,y1,x2,y2],
        'rows': List[List[text_region]],
        'columns': List[List[text_region]],
        'confidence': float
    }

"""

from collections import defaultdict
import cv2
import numpy as np
import re

class TextPositionTableDetector:
    def __init__(self):
        """
        Table detector for borderless tables based on text position analysis
        """
        pass
    
    def _extract_text_regions(self, image_path, min_quality_score=0.3, method='mser'):
        """
        Extract text regions with configurable method selection to reduce computation
        
        Args:
            image_path: path to input image
            min_quality_score: minimum confidence threshold for filtering
            method: 'mser', 'contour', 'morph', or 'all' for detection method
            
        Returns:
            list of text regions with bbox and basic info
        """
        print(f"Extracting text regions from: {image_path}")
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Select detection method based on parameter
        if method == 'mser':
            print("Using MSER method only...")
            all_regions = self._extract_text_mser(gray)
        elif method == 'contour':
            print("Using contour method only...")
            all_regions = self._extract_text_contours(gray)
        elif method == 'morph':
            print("Using morphological method only...")
            all_regions = self._extract_text_morphological(gray)
        else:  # method == 'all'
            print("Using all detection methods...")
            mser_regions = self._extract_text_mser(gray)
            morph_regions = self._extract_text_morphological(gray)
            contour_regions = self._extract_text_contours(gray)
            all_regions = mser_regions + morph_regions + contour_regions
        
        print(f"Detected: {len(all_regions)} regions before filtering")
        
        # Apply quality filtering using passed parameter
        filtered_regions = self._filter_regions_by_quality(all_regions, min_quality_score)
        
        print(f"Final result: {len(filtered_regions)} text regions after quality filtering")
        return filtered_regions

    def extract_text_regions(self, image_path, min_quality_score=0.3, method='mser'):
        """Public wrapper around :meth:`_extract_text_regions`."""
        return self._extract_text_regions(image_path, min_quality_score, method)
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
    
    def _filter_overlapping_mser_regions_spatial(self, mser_regions, overlap_threshold=0.3, grid_size=100):
        """
        Fast MSER overlap filtering using spatial grid to find nearby regions first
        
        Args:
            mser_regions: list of MSER detected regions
            overlap_threshold: minimum overlap ratio to consider as overlapping
            grid_size: pixel size of spatial grid cells
            
        Returns:
            filtered list of non-overlapping regions or merged text regions
        """
        if len(mser_regions) <= 1:
            return mser_regions
        
        print(f"Spatial filtering overlapping MSER regions from {len(mser_regions)} regions...")
        
        # Step 1: Build spatial hash map
        spatial_grid = defaultdict(list)
        
        for region in mser_regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Calculate which grid cells this region occupies
            grid_x1, grid_y1 = int(x1 // grid_size), int(y1 // grid_size)
            grid_x2, grid_y2 = int(x2 // grid_size), int(y2 // grid_size)
            
            # Add region to all grid cells it overlaps
            for gx in range(grid_x1, grid_x2 + 1):
                for gy in range(grid_y1, grid_y2 + 1):
                    spatial_grid[(gx, gy)].append(region)
        
        print(f"Built spatial grid with {len(spatial_grid)} occupied cells")
        
        # Step 2: Process each region only against its spatial neighbors
        processed_regions = set()
        merged_regions = []
        removed_regions = set()
        
        for region in mser_regions:
            if region['id'] in removed_regions:
                continue
                
            if region['id'] in processed_regions:
                continue
            
            # Find spatial neighbors
            neighbors = self._find_spatial_neighbors(region, spatial_grid, grid_size)
            
            # Check overlaps only with neighbors
            overlapping_neighbors = []
            for neighbor in neighbors:
                if neighbor['id'] != region['id'] and neighbor['id'] not in removed_regions:
                    overlap_ratio = self._calculate_overlap_ratio(region['bbox'], neighbor['bbox'])
                    if overlap_ratio > overlap_threshold:
                        overlapping_neighbors.append(neighbor)
            
            if overlapping_neighbors:
                # Check if overlapping neighbors form a text line
                line_regions = [region] + overlapping_neighbors
                if self._regions_form_text_line(line_regions):
                    # Merge into single text region
                    merged_region = self._merge_multiple_regions(line_regions)
                    merged_regions.append(merged_region)
                    
                    # Mark all original regions as processed
                    for r in line_regions:
                        processed_regions.add(r['id'])
                else:
                    # Remove all as non-text overlaps
                    for r in line_regions:
                        removed_regions.add(r['id'])
            else:
                # No overlaps, keep the region
                merged_regions.append(region)
                processed_regions.add(region['id'])
        
        print(f"Spatial filtering complete: kept {len(merged_regions)} regions "
              f"(removed {len(mser_regions) - len(merged_regions)})")
        
        return merged_regions
    
    def _find_spatial_neighbors(self, region, spatial_grid, grid_size):
        """
        Find spatial neighbors of a region using the grid
        """
        x1, y1, x2, y2 = region['bbox']
        
        # Calculate grid range to check (include adjacent cells)
        grid_x1, grid_y1 = int(x1 // grid_size) - 1, int(y1 // grid_size) - 1
        grid_x2, grid_y2 = int(x2 // grid_size) + 1, int(y2 // grid_size) + 1
        
        neighbors = []
        
        for gx in range(grid_x1, grid_x2 + 1):
            for gy in range(grid_y1, grid_y2 + 1):
                if (gx, gy) in spatial_grid:
                    neighbors.extend(spatial_grid[(gx, gy)])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_neighbors = []
        for neighbor in neighbors:
            if neighbor['id'] not in seen:
                unique_neighbors.append(neighbor)
                seen.add(neighbor['id'])
        
        return unique_neighbors
    
    def _regions_form_text_line(self, regions):
        """
        Check if overlapping regions could form a coherent text line
        
        Args:
            regions: list of overlapping regions
            
        Returns:
            True if regions form a text line
        """
        if len(regions) < 2:
            return True
        
        # Sort regions by X coordinate
        x_sorted = sorted(regions, key=lambda r: r['center'][0])
        
        # Check if regions are roughly aligned horizontally (same line)
        y_coords = [r['center'][1] for r in regions]
        y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
        
        # Calculate average height
        heights = [r['bbox'][3] - r['bbox'][1] for r in regions]
        avg_height = np.mean(heights)
        
        # Text line criteria:
        # 1. Low Y variance (horizontally aligned)
        # 2. Reasonable horizontal spacing
        # 3. Similar heights
        
        # Check horizontal alignment
        if y_variance > (avg_height * 0.5) ** 2:  # Too much vertical variation
            return False
        
        # Check if regions are reasonably spaced horizontally
        for i in range(len(x_sorted) - 1):
            curr_region = x_sorted[i]
            next_region = x_sorted[i + 1]
            
            curr_right = curr_region['bbox'][2]
            next_left = next_region['bbox'][0]
            
            gap = next_left - curr_right
            max_expected_gap = avg_height * 2  # Reasonable character spacing
            
            if gap > max_expected_gap:  # Too far apart
                return False
        
        # Check height consistency
        height_ratios = [min(heights) / max(heights)] if len(heights) > 1 else [1.0]
        if height_ratios[0] < 0.5:  # Heights too different
            return False
        
        return True
    
    def _merge_multiple_regions(self, regions):
        """
        Merge multiple overlapping regions into one
        
        Args:
            regions: list of regions to merge
            
        Returns:
            merged region
        """
        # Calculate merged bounding box
        x1 = min(r['bbox'][0] for r in regions)
        y1 = min(r['bbox'][1] for r in regions)
        x2 = max(r['bbox'][2] for r in regions)
        y2 = max(r['bbox'][3] for r in regions)
        
        # Use highest confidence
        max_confidence = max(r['confidence'] for r in regions)
        
        merged_region = {
            'id': f"merged_{'_'.join(r['id'] for r in regions[:3])}",  # Limit ID length
            'bbox': [x1, y1, x2, y2],
            'text': f"merged_text_line",
            'confidence': max_confidence,
            'center': [(x1 + x2)/2, (y1 + y2)/2],
            'method': 'mser_line_merged'
        }
        
        return merged_region
    
    def _extract_text_mser(self, gray_image):
        """
        Extract text regions using MSER with spatial-based overlap filtering
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
        
        # Use spatial grid-based overlap filtering
        filtered_regions = self._filter_overlapping_mser_regions_spatial(text_regions, overlap_threshold=0.3)
        
        return filtered_regions
    
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
        text_regions = self._deduplicate_text_regions_fast(all_regions)
        print(f"  After deduplication: {len(text_regions)} regions")
        
        if save_debug_images:
            # Save individual method results
            self._save_method_debug(img, mser_regions, 'debug_mser_regions.jpg', 'MSER')
            self._save_method_debug(img, morph_regions, 'debug_morph_regions.jpg', 'Morphological')
            self._save_method_debug(img, contour_regions, 'debug_contour_regions.jpg', 'Contour')
            self._save_method_debug(img, text_regions, 'debug_all_text_regions.jpg', 'All Combined')
        
        # Step 2: Analyze alignment
        alignment_info = self._analyze_text_alignment_fast(text_regions, alignment_tolerance=20)
        
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
            final_img = self._visualize_text_based_tables(image_path, tables, text_regions)
        
        return {
            'text_regions': text_regions,
            'alignment_info': alignment_info,
            'tables': tables,
            'method_breakdown': {
                'mser': len(mser_regions),
                'morphological': len(morph_regions),
                'contour': len(contour_regions),
                'final': len(text_regions)
            }
        }
    
    def _save_method_debug(self, img, regions, filename, method_name):
        """Save debug image for a specific detection method."""
        debug_img = img.copy()

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

    def assign_text_regions_to_frames(self, frames, text_regions, method='center', overlap_threshold=0.1):
        """Assign text regions to detected table frames.

        Args:
            frames: iterable of table/frame dicts containing at least a ``bbox`` and optional ``id``.
            text_regions: iterable of text-region dicts (as produced by detection methods in this class).
            method: 'center' to use the text center point, 'overlap' to use IOU-style overlap.
            overlap_threshold: minimum overlap ratio when ``method='overlap'``.

        Returns:
            tuple(assignments, unassigned)
                assignments: List of dicts {'frame_id', 'bbox', 'frame', 'text_regions'}
                unassigned: List of text regions that didn't match any frame
        """
        assignments = []

        for idx, frame in enumerate(frames):
            bbox = frame.get('bbox')
            if bbox is None:
                raise ValueError("Each frame must provide a 'bbox' field [x1, y1, x2, y2].")

            frame_id = frame.get('id', f'frame_{idx}')
            assignments.append({
                'frame_id': frame_id,
                'bbox': bbox,
                'frame': frame,
                'text_regions': []
            })

        unassigned = []

        for region in text_regions:
            region_bbox = region.get('bbox')
            region_center = region.get('center')
            matched = False

            for entry in assignments:
                x1, y1, x2, y2 = entry['bbox']

                if method == 'center':
                    if region_center is None:
                        continue
                    cx, cy = region_center
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        entry['text_regions'].append(region)
                        matched = True
                        break
                else:
                    if region_bbox is None:
                        continue
                    overlap = self._calculate_overlap_ratio(region_bbox, entry['bbox'])
                    if overlap >= overlap_threshold:
                        entry['text_regions'].append(region)
                        matched = True
                        break

            if not matched:
                unassigned.append(region)

        return assignments, unassigned

    def analyze_text_within_frames(self, frames, text_regions, alignment_tolerance=20,
                                   min_regions=2, method='center', overlap_threshold=0.1):
        """Analyze text alignment inside each frame to reduce global complexity.

        Args:
            frames: list of table/frame dicts (e.g., from WiredTableDetector).
            text_regions: list of text regions (e.g., from _extract_text_regions).
            alignment_tolerance: tolerance passed to ``_analyze_text_alignment_fast``.
            min_regions: minimum regions required to trigger alignment analysis per frame.
            method: assignment strategy ('center' or 'overlap').
            overlap_threshold: overlap threshold when using the 'overlap' method.

        Returns:
            dict with keys:
                'frames': List of per-frame analysis dicts containing:
                    'frame_id', 'bbox', 'text_count', 'text_regions', 'alignment'
                'unassigned_regions': text regions not matched to any frame
        """
        assignments, unassigned = self.assign_text_regions_to_frames(
            frames, text_regions, method=method, overlap_threshold=overlap_threshold
        )

        results = []

        for entry in assignments:
            frame_regions = entry['text_regions']
            analysis = {
                'frame_id': entry['frame_id'],
                'bbox': entry['bbox'],
                'frame': entry['frame'],
                'text_regions': frame_regions,
                'text_count': len(frame_regions),
                'alignment': {'rows': [], 'columns': []}
            }

            if len(frame_regions) >= min_regions:
                alignment = self._analyze_text_alignment_fast(frame_regions, alignment_tolerance)
                analysis['alignment'] = alignment
                analysis['row_count'] = len(alignment['rows'])
                analysis['column_count'] = len(alignment['columns'])
            else:
                analysis['row_count'] = 0
                analysis['column_count'] = 0

            results.append(analysis)

        return {
            'frames': results,
            'unassigned_regions': unassigned
        }
    
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

# Usage example
def detect_borderless_tables_simple(image_path):
    """
    Simple function for borderless table detection
    """
    detector = TextPositionTableDetector()
    return detector.detect_borderless_tables(image_path)

# Test function
if __name__ == "__main__":
    image_path = "/Users/bingzhi/git/table_detection/page_0.jpg"
    
    # Detect borderless tables
    tables = detect_borderless_tables_simple(image_path)
    
    print(f"\nDetected {len(tables)} borderless tables in total.")