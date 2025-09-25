from paddleocr import PaddleOCR

from PIL import Image
import cv2
import numpy as np



def basic_ocr(image_path, lang='en', output_file=None):
    """
    Basic PaddleOCR functionality

    Args:
        image_path: Path to input image
        lang: Language ('en', 'ch', etc.)xq
        output_file: Optional output file to save text

    Returns:
        extracted_text: String of all extracted text
    """
    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    # Perform OCR
    result = ocr.ocr(image_path, cls=True)

    # Extract text
    extracted_text = ""
    if result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            print(f"Text: {text}, Confidence: {confidence:.4f}")
            extracted_text += text + "\n"

    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Results saved to {output_file}")

    return extracted_text.strip()

def visualize_ocr(image_path, lang='en', save_path=None):
    """
    OCR with visualization

    Args:
        image_path: Path to input image
        lang: Language
        save_path: Optional path to save visualization
    """
    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    # Perform OCR
    result = ocr.ocr(image_path, cls=True)

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Extract boxes, texts, and scores
    if result[0]:
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        # Draw OCR results
        im_show = draw_ocr(image, boxes, texts, scores, font_path=None)
        im_show = Image.fromarray(im_show)

        # Save if specified
        if save_path:
            im_show.save(save_path)
            print(f"Visualization saved to {save_path}")

        return im_show, texts

    return image, []

if __name__ == "__main__":
    # Example usage
    image_path = "small.png"

    # Basic OCR
    print("=== Basic OCR ===")
    text = basic_ocr(image_path, lang='en', output_file='ocr_output.txt')
    print(f"\nExtracted text:\n{text}")

    # OCR with visualization
    print("\n=== OCR with Visualization ===")
    vis_image, texts = visualize_ocr(image_path, lang='en', save_path='ocr_visualization.png')
    print(f"Found {len(texts)} text regions")