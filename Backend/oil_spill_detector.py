import cv2
import numpy as np

def detect_oil_spill(image):
    """
    Detect oil spill regions in the given image.
    Returns a tuple: (annotated_image, area_km2, pixel_count, confidence, risk_level).
    - annotated_image: NumPy array (BGR) with detected spill area highlighted.
    - area_km2: Estimated area of the spill in square kilometers.
    - pixel_count: Number of pixels detected as oil spill.
    - confidence: Confidence level (0.0 to 1.0) of the detection.
    - risk_level: Risk level string ("Low", "Medium", "High", or "None").
    """
    # Convert image to grayscale for easier analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to identify very dark regions (potential oil spills)
    # Pixels below intensity 30 (on 0-255 scale) are considered part of an oil spill
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    # Find contours of the thresholded regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate pixel count of the spill region
    pixel_count = int(cv2.countNonZero(mask))
    total_pixels = image.shape[0] * image.shape[1]
    # Estimate area in square kilometers (assuming 1 pixel ~ 1 m^2 for demonstration purposes)
    area_km2 = pixel_count / 1_000_000.0  # 1e6 pixels ≈ 1 km^2
    # Determine risk level based on the size of the spill
    if pixel_count == 0:
        risk_level = "None"
    elif pixel_count > 100000:  # large spill
        risk_level = "High"
    elif pixel_count > 5000:    # moderate spill
        risk_level = "Medium"
    else:                       # small spill
        risk_level = "Low"
    # Determine confidence level (simple heuristic based on spill area proportion)
    if pixel_count == 0:
        confidence = 0.0
    else:
        ratio = pixel_count / float(total_pixels)
        if ratio > 0.05:       # spill covers >5% of image
            confidence = 0.95
        elif ratio > 0.02:     # 2-5% of image
            confidence = 0.85
        elif ratio > 0.005:    # 0.5-2% of image
            confidence = 0.70
        else:                  # very small area
            confidence = 0.50
    # Create a copy of the original image to draw annotations
    if contours:
        # Draw filled contour (overlay) in red on a copy of the image
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)  # fill contours with red
        # Blend the overlay with the original image to get semi-transparent fill
        annotated_image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        # Draw contour outlines in red on the blended image for clear edges
        cv2.drawContours(annotated_image, contours, -1, (0, 0, 255), 2)
    else:
        # If no spill detected, annotated image is just a copy of original
        annotated_image = image.copy()
    return annotated_image, area_km2, pixel_count, confidence, risk_level
