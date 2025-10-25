from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from oil_spill_detector import detect_oil_spill

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def parse_ais_file(ais_file):
    """Parse the uploaded AIS file (CSV or text) and return a list of vessel descriptions."""
    try:
        content = ais_file.read()
        if not content:
            return []
        text = content.decode('utf-8', errors='ignore')
    except Exception as e:
        print("AIS file read error:", e)
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header_line = lines[0]
    header_lower = header_line.lower()
    # Determine if the first line is a header by checking common keywords
    header_terms = ["mmsi", "name", "vessel", "latitude", "longitude"]
    has_header = any(term in header_lower for term in header_terms)
    vessels = []
    if has_header:
        # Determine delimiter (comma or tab)
        delimiter = ',' if header_line.count(',') >= header_line.count('\t') else '\t'
        headers = [h.strip() for h in header_line.split(delimiter)]
        # Create index lookup for relevant fields
        index_map = {h.lower(): idx for idx, h in enumerate(headers)}
        name_idx = type_idx = lat_idx = lon_idx = mmsi_idx = None
        for key, idx in index_map.items():
            if 'name' in key and name_idx is None:
                name_idx = idx
            if 'type' in key and type_idx is None:
                type_idx = idx
            if key in ('latitude', 'lat') and lat_idx is None:
                lat_idx = idx
            if key in ('longitude', 'lon', 'long') and lon_idx is None:
                lon_idx = idx
            if key == 'mmsi' and mmsi_idx is None:
                mmsi_idx = idx
        # Parse each vessel entry line (skip header line)
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(delimiter)]
            if not parts or parts[0] == '':
                continue
            name = parts[name_idx] if name_idx is not None and name_idx < len(parts) else ''
            vessel_type = parts[type_idx] if type_idx is not None and type_idx < len(parts) else ''
            lat = parts[lat_idx] if lat_idx is not None and lat_idx < len(parts) else ''
            lon = parts[lon_idx] if lon_idx is not None and lon_idx < len(parts) else ''
            mmsi = parts[mmsi_idx] if mmsi_idx is not None and mmsi_idx < len(parts) else ''
            # Build a description string for the vessel
            desc = ""
            if name:
                desc += name
            elif mmsi:
                desc += f"MMSI {mmsi}"
            if vessel_type:
                desc += f" ({vessel_type})"
            if lat and lon:
                desc += f" at {lat},{lon}"
            if desc:
                vessels.append(desc)
    else:
        # No header present: return raw lines as vessel entries
        for line in lines:
            vessels.append(line)
    return vessels

@app.route('/predict', methods=['POST'])
def predict():
    # Check and retrieve uploaded files
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files['image']
    ais_file = request.files.get('ais')
    # Read the image file into an OpenCV image
    file_bytes = image_file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_image is None:
        return jsonify({"error": "Invalid image file"}), 400
    # Perform oil spill detection on the image
    try:
        annotated_img, area, pixel_count, confidence, risk_level = detect_oil_spill(cv_image)
    except Exception as e:
        print("Detection error:", e)
        return jsonify({"error": "Failed to process image"}), 500
    # Encode the annotated image to base64 string
    _, buffer = cv2.imencode('.png', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    # Parse AIS file if provided
    vessels = []
    if ais_file and ais_file.filename:
        ais_file.seek(0)  # reset file pointer
        try:
            vessels = parse_ais_file(ais_file)
        except Exception as e:
            print("AIS parsing error:", e)
            vessels = []
    # Prepare JSON response with results
    result = {
        "area": round(area, 4),              # area in square km (approximate)
        "pixel_count": int(pixel_count),     # number of pixels identified as spill
        "confidence": round(confidence, 2),  # confidence level (0-1)
        "risk_level": risk_level,            # risk level as a string
        "vessels": vessels,                  # list of nearby vessels or from AIS data
        "image": img_base64                  # base64-encoded annotated image
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
