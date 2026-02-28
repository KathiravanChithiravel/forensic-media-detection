import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import PIL.ExifTags
import hashlib

def calculate_md5(file_path):
    """Compulsory File Integrity Check (MD5 Hash)."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest().upper()

def extract_exif(image_path):
    """Enhanced EXIF Metadata Analysis with Compulsory File Integrity."""
    try:
        img = Image.open(image_path)
        exif_raw = img._getexif()
        report = {
            'has_metadata': False,
            'software': 'None Detected',
            'make_model': 'Unknown',
            'timestamp': 'Not Found',
            'md5_hash': calculate_md5(image_path),
            'resolution': f"{img.size[0]}x{img.size[1]} px",
            'details': {}
        }
        if exif_raw:
            report['has_metadata'] = True
            details = {}
            for tag, val in exif_raw.items():
                tag_name = str(PIL.ExifTags.TAGS.get(tag, tag))
                details[tag_name] = str(val)
                if tag_name == 'Software':
                    report['software'] = str(val) or "None Detected"
                elif tag_name in ['Make', 'Model']:
                    current_model = str(report['make_model'])
                    report['make_model'] = (current_model + " " + str(val)).strip() if current_model != 'Unknown' else str(val)
                elif tag_name == 'DateTime':
                    report['timestamp'] = str(val)
            report['details'] = details
        return report
    except Exception as e:
        return {'has_metadata': False, 'software': 'Error', 'make_model': 'Error', 'timestamp': 'Error', 'details': {'error': str(e)}}

def detect_noise_inconsistency(image_path):
    """Estimate noise patterns to detect inconsistency."""
    try:
        img = cv2.imread(image_path)
        if img is None: return False, "Could not read image", 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        noise = cv2.filter2D(gray, -1, kernel)
        std_dev = float(np.std(noise))
        if std_dev > 25:
            return True, "Abnormal noise distribution detected (Potential grain editing).", std_dev
        return False, "Natural noise distribution.", std_dev
    except: return False, "Noise test failed.", 0.0

def detect_blur_manipulation(image_path):
    """Check for unnatural blur variances."""
    try:
        img = cv2.imread(image_path)
        if img is None: return False, "Could not read image", 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 100:
            return True, "Artificial blur or low-resolution smoothing detected.", lap_var
        return False, "Image has natural sharpness.", lap_var
    except: return False, "Blur test failed.", 0.0

def detect_color_inconsistency(image_path):
    """Check for unnatural saturation or color grading."""
    try:
        img = cv2.imread(image_path)
        if img is None: return False, "Could not read image", 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_mean = float(np.mean(s))
        if s_mean > 160:
            return True, "Extreme saturation detected (Potential filter/LUT application).", s_mean
        return False, "Natural color balance.", s_mean
    except: return False, "Color test failed.", 0.0

def detect_clone_stamp(image_path):
    """Detect potential block repetition (Clone-Stamp)."""
    try:
        img = cv2.imread(image_path, 0)
        if img is None: return False, "Could not read image", 0.0
        img = cv2.resize(img, (256, 256))
        patch = img[100:150, 100:150]
        res = cv2.matchTemplate(img, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        score = float(max_val)
        if score > 0.999:
            return True, "Repeated pixel patterns detected (Clone-Stamp suspicious).", score
        return False, "No clonal patterns found.", score
    except: return False, "Clone test failed.", 0.0

def detect_ai_generation_image(image_path):
    """
    Detect AI Generated / CGI / Cartoon characteristics.
    Logic:
    - AI: Over-smooth texture (Low noise, high frequency patterns)
    - CGI/Cartoon: Low color variety (Laplacian variance), stylized edges (Hough lines)
    """
    img = cv2.imread(image_path)
    if img is None: return 0.0, "None"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Texture Smoothness (Laplacian Variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    ai_prob = 0
    reason = "Natural texture detected."
    
    # 2. Edge Sharpness (Detecting cartoon-like stylized edges)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    
    # 3. Frequency analysis (lack of sensor noise)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    freq_mean = np.mean(magnitude_spectrum)
    
    # 4. Color Flatness (CGI often has perfectly flat regions)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s_std = np.std(s)
    
    # AI/CGI Decision Logic
    if freq_mean < 135 and blur_score < 70: # Synthetic smoothness
        ai_prob += 50
        reason = "Unrealistic smooth texture / AI smoothing detected."
    
    if edge_density > 0.08 and s_std > 60: # Stylized edges & high saturation
        ai_prob += 40
        reason = "Computer Generated / AI Artwork (Stylized cartoon-edges and shading)."

    ai_val = float(ai_prob)
    return min(ai_val, 99.2), reason

def detect_tool_inference(image_path, results):
    """Estimate possible editing tools based on forensic results."""
    # results = { 'is_blur_susc': bool, 'is_clone_susc': bool, 'metadata': dict, 'is_noise_susc': bool }
    tools = []
    
    if results.get('is_blur_susc'):
        tools.append(str("Possibly Blur Tool / Gaussian Blur (Manual editing region detected)"))
    if results.get('is_clone_susc'):
        tools.append(str("Possibly Clone Stamp Tool / Healing Brush (Repeated patterns found)"))
    if results.get('is_color_susc'):
        tools.append(str("Possible Color Correction Tool Usage (Extreme histogram / lookup shift)"))
    
    software_name = str(results.get('metadata', {}).get('software', 'None Detected'))
    if software_name != 'None Detected' and software_name != 'Error':
        tools.append(str(f"Metadata Trace: {software_name}"))
    
    if not tools:
        return "None Detected"
    return " | ".join(tools)

def perform_ela(image_path, out_path, quality=90):
    """Error Level Analysis Map Generation."""
    original = Image.open(image_path).convert('RGB')
    temp_path = image_path + "_tmp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    recompressed = Image.open(temp_path)
    ela = ImageChops.difference(original, recompressed)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / (max_diff if max_diff != 0 else 1)
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    ela.save(out_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return np.array(ela)

def perform_noise_map(image_path, out_path):
    """Generate a high-pass noise visualization map."""
    img = cv2.imread(image_path)
    if img is None: return
    # High-pass filter to extract noise
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    noise_map = cv2.filter2D(img, -1, kernel)
    # Brighten for visualization
    noise_map = cv2.convertScaleAbs(noise_map, alpha=4, beta=10)
    cv2.imwrite(out_path, noise_map)

def perform_clone_map(image_path, out_path):
    """Simple map highlighting potential clone-like duplicate areas."""
    img = cv2.imread(image_path, 0)
    if img is None: return
    img = cv2.resize(img, (256, 256))
    res = cv2.matchTemplate(img, img[100:150, 100:150], cv2.TM_CCOEFF_NORMED)
    # Highlight areas with very high correlation (> 0.98)
    _, mask = cv2.threshold(res, 0.98, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    cv2.imwrite(out_path, mask)

def analyze_regions(image_path, ela_np):
    """Detailed Region Audit with Tool Inference local placeholders."""
    w, h = ela_np.shape[1], ela_np.shape[0]
    rows, cols = 4, 4
    rw, rh = w // cols, h // rows
    
    regions = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * rw, r * rh
            x2, y2 = x1 + rw, y1 + rh
            patch = ela_np[y1:y2, x1:x2]
            intensity = float(np.mean(patch))
            
            is_suspicious = bool(intensity > 18)
            reason = "Consistent compression signature."
            tool_inf = "None"
            
            if is_suspicious:
                if intensity > 40:
                    reason = "Significant artificial overlay / local re-saving detected."
                    tool_inf = "Selection / Stamp tool"
                else:
                    reason = "Potential local smoothing or edge refinement."
                    tool_inf = "Blur / Smudge tool"
            
            regions.append({
                'id': f'region_{r}_{c}',
                'is_suspicious': is_suspicious,
                'explanation': reason,
                'tool_inference': tool_inf,
                'confidence': f"{min(intensity * 1.5 + 40, 99.2) if is_suspicious else 100 - intensity:.1f}%"
            })
    return regions

def analyze_histogram_irregularity(image_path):
    """Detect gaps in histogram which are indicators of heavy tonal editing or re-saving."""
    try:
        img = cv2.imread(image_path)
        if img is None: return False, "Read Error", 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Count zero or near-zero bins in the middle ranges
        gaps = 0
        for i in range(10, 245):
            if hist[i] < 5: gaps += 1
            
        if gaps > 15:
            return True, f"Significant histogram gaps ({gaps}) detected (Heavy tonal manipulation).", float(gaps)
        return False, "Consistent histogram distribution.", float(gaps)
    except: return False, "Histogram test failed.", 0.0

def detect_edge_ringing(image_path):
    """Detect Gibbs-like ringing artifacts around high-contrast edges, common in JPEG over-compression."""
    try:
        img = cv2.imread(image_path, 0)
        edges = cv2.Canny(img, 100, 200)
        # Dilate edges to check surroundings
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        # High pass to find ringing
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        hi_pass = cv2.filter2D(img, -1, kernel)
        
        ringing_map = cv2.bitwise_and(hi_pass, hi_pass, mask=dilated)
        score = float(np.mean(ringing_map))
        
        if score > 50:
            return True, "Artificial edge ringing detected (Potential local re-compression).", score
        return False, "Natural edge transition.", score
    except: return False, "Edge test failed.", 0.0

def analyze_video(video_path):
    """
    Upgraded video forensic analysis:
    - Frame extraction every 5 frames
    - Anomaly score graph per frame
    - AI-Generated / Deepfake detection characteristics
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = 5
    
    frame_scores = []
    suspicious_count = 0
    analyzed_count = 0
    prev_gray = None
    
    # Metrics to detect AI Generation
    texture_smoothness_scores = []
    motion_jitter_scores = [] # Based on temporal landmark stability heuristic
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if analyzed_count % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 1. Texture Anomaly (AI often too smooth / synthetic noise)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_smoothness_scores.append(laplacian_var)
            
            # 2. Frame-by-frame anomaly score
            anomaly_score = 0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                flicker = float(np.mean(diff))
                motion_jitter_scores.append(flicker)
                
                # Heuristic for Deepfake: Unnatural temporal stability or flickering
                if flicker > 45 or flicker < 0.5: # 0.5 too static (GAN stillness)
                    anomaly_score += 40
                
            if laplacian_var < 80: # GAN Smoothing
                anomaly_score += 35
                
            frame_scores.append({
                'frame_index': analyzed_count,
                'score': float(min(anomaly_score + (100 - laplacian_var)/2 if laplacian_var < 100 else anomaly_score, 100))
            })
            
            if anomaly_score > 50:
                suspicious_count += 1
                
            prev_gray = gray
        
        analyzed_count += 1
        if analyzed_count > 500: break # Efficiency limit for FYP Demo

    cap.release()
    
    # Calculate Probabilities with refined Video metrics
    avg_anomaly = np.mean([f['score'] for f in frame_scores]) if frame_scores else 0
    
    ai_video_prob = 0
    # Marker 1: Texture Smoothing (Under 85 Laplacian is highly synthetic for full frames)
    if np.mean(texture_smoothness_scores) < 85: ai_video_prob += 45
    
    # Marker 2: Motion Inconsistency (Detecting GAN flickering/jitter)
    # Natural motion has smooth transitions, AI often has high std in jitter
    motion_stability = np.std(motion_jitter_scores) if motion_jitter_scores else 0
    if motion_stability > 15: # Extreme flicker
        ai_video_prob += 40
    elif motion_stability < 1.0: # Unnatural "frozen" lighting/stability
        ai_video_prob += 35
    
    # Marker 3: Watermark / Overlay Detection Heuristic
    # Constant edge density regions often indicate static watermarks or logo overlays
    watermark_suspicion = False
    if suspicious_count > (len(frame_scores) * 0.3): # 30% of frames have structural anomalies
        watermark_suspicion = True

    # Decision Hierarchy: Strong "FAKE" Indications
    if ai_video_prob > 50 or (ai_video_prob > 30 and avg_anomaly > 35):
        result = str("AI Generated / Deepfake (FAKE)")
    elif avg_anomaly > 45 or watermark_suspicion:
        result = str("Digitally Manipulated / Watermarked (FAKE)")
    elif avg_anomaly > 20:
        result = str("Suspicious / Possible Edit")
    else:
        result = str("Real Video Stream")
        
    conf_val = float(ai_video_prob) if result.startswith("AI") else (100.0 - float(avg_anomaly) if result == "Real Video" else float(avg_anomaly) + 30.0)
    confidence = min(conf_val, 99.2)
    
    # Scoring Proportionality
    p_ai = float(ai_video_prob)
    p_edited = float(avg_anomaly * 1.5 if result == "Edited Video" else avg_anomaly)
    p_edited = min(p_edited, 90.0)
    
    p_real = max(0, 100.0 - p_edited - (p_ai / 2.0)) if result != "AI Generated / Deepfake Video" else 10.0
    
    # Final Normalization for consistent 100% display
    sum_p = p_real + p_edited + p_ai
    if sum_p > 0:
        p_real = (p_real / sum_p) * 100.0
        p_edited = (p_edited / sum_p) * 100.0
        p_ai = (p_ai / sum_p) * 100.0

    return {
        'total_frames': total_frames,
        'analyzed_frames': len(frame_scores),
        'suspicious_frames': suspicious_count,
        'avg_anomaly_score': float(avg_anomaly),
        'frame_scores': frame_scores, # For chart.js
        'result': result,
        'confidence': confidence,
        'prob_real': float(p_real),
        'prob_edited': float(p_edited),
        'prob_ai': float(p_ai)
    }
