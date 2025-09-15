import cv2
import numpy as np
import imutils
import sys

def resize(img, width=1000):
    h, w = img.shape[:2]
    if w > width:
        ratio = width / float(w)
        return cv2.resize(img, (width, int(h*ratio)), interpolation=cv2.INTER_AREA)
    return img

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document_contour(img_gray):
    # edge detection then find biggest 4-point contour
    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    edged = cv2.Canny(blurred, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def detect_card_orientation(img):
    """
    Analyze the distribution of pixels and edges to determine the correct orientation 
    of an ID card. Italian ID cards typically have:
    1. More features at the top (photo, headers, etc.)
    2. More horizontal lines at the top
    3. More empty space at the bottom
    4. Different pixel intensity distribution between top and bottom
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    h, w = gray.shape
    
    # Check if portrait or landscape
    is_portrait = h > w
    
    # First, ensure we're in landscape mode
    working_img = img.copy()
    if is_portrait:
        working_img = cv2.rotate(working_img, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
    
    # Method 1: Analyze edge distribution (top vs bottom)
    edges = cv2.Canny(gray, 50, 150)
    
    # Split image into top and bottom halves
    top_half = edges[:h//2, :]
    bottom_half = edges[h//2:, :]
    
    # Count non-zero pixels (edges) in each half
    top_edges = cv2.countNonZero(top_half)
    bottom_edges = cv2.countNonZero(bottom_half)
    
    # Method 2: Analyze horizontal lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                        minLineLength=w//10, maxLineGap=20)
    
    top_horizontal_lines = 0
    bottom_horizontal_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle
            if abs(x2 - x1) > 0:  # Avoid division by zero
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Consider horizontal lines (angle close to 0 or 180)
                if angle < 20 or angle > 160:
                    if y1 < h//2 and y2 < h//2:
                        top_horizontal_lines += 1
                    elif y1 >= h//2 and y2 >= h//2:
                        bottom_horizontal_lines += 1
    
    # Method 3: Analyze vertical lines (ID cards typically have more vertical lines on the left - photo side)
    left_vertical_lines = 0
    right_vertical_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle
            if abs(x2 - x1) > 0:  # Avoid division by zero
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Consider vertical lines (angle close to 90)
                if 70 < angle < 110:
                    if x1 < w//2 and x2 < w//2:
                        left_vertical_lines += 1
                    elif x1 >= w//2 and x2 >= w//2:
                        right_vertical_lines += 1
    
    # Method 4: Analyze pixel intensity distribution
    top_intensity = np.mean(gray[:h//2, :])
    bottom_intensity = np.mean(gray[h//2:, :])
    
    # Method 5: Analyze variation in pixel intensity (standard deviation)
    top_std = np.std(gray[:h//2, :])
    bottom_std = np.std(gray[h//2:, :])
    
    # Make decision based on combined analysis
    needs_180_rotation = False
    
    # Score system for orientation determination
    orientation_score = 0
    
    # If top half has more edges, likely correctly oriented (positive score)
    if top_edges > bottom_edges:
        orientation_score += 2
    else:
        orientation_score -= 2
    
    # If top half has more horizontal lines, likely correctly oriented
    if top_horizontal_lines > bottom_horizontal_lines:
        orientation_score += 1
    else:
        orientation_score -= 1
    
    # If left half has more vertical lines (photo side), likely correctly oriented
    if left_vertical_lines > right_vertical_lines:
        orientation_score += 1
    else:
        orientation_score -= 1
    
    # ID cards typically have more pixel variation (text, photo) in the top half
    if top_std > bottom_std:
        orientation_score += 1
    else:
        orientation_score -= 1
    
    # If score is negative, we need to rotate 180 degrees
    if orientation_score < 0:
        needs_180_rotation = True
    
    # Debug information
    print(f"Orientamento automatico:")
    print(f"  Bordi: superiore {top_edges}, inferiore {bottom_edges}")
    print(f"  Linee orizzontali: superiore {top_horizontal_lines}, inferiore {bottom_horizontal_lines}")
    print(f"  Linee verticali: sinistra {left_vertical_lines}, destra {right_vertical_lines}")
    print(f"  Variazione pixel: superiore {top_std:.2f}, inferiore {bottom_std:.2f}")
    print(f"  Punteggio orientamento: {orientation_score} ({'ruota 180°' if orientation_score < 0 else 'corretto'})")
    
    return working_img, needs_180_rotation, is_portrait

def correct_orientation(img, manual_rotation=None):
    """
    Detect and correct the orientation of an ID card to ensure it's horizontal
    with text in the correct reading direction.
    
    Parameters:
        img (numpy.ndarray): Input image
        manual_rotation (int, optional): Override automatic detection with manual rotation
                                        0 = no rotation
                                        1 = 90° clockwise
                                        2 = 180°
                                        3 = 270° clockwise (90° counterclockwise)
    """
    # If manual rotation is specified, apply it directly
    if manual_rotation is not None:
        if manual_rotation == 0:
            return img
        elif manual_rotation == 1:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif manual_rotation == 2:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif manual_rotation == 3:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"Valore rotazione manuale non valido: {manual_rotation}. Utilizzo rilevamento automatico.")
    
    # Automatic orientation detection and correction
    working_img, needs_180_rotation, was_portrait = detect_card_orientation(img)
    
    # If needed, rotate 180 degrees to get the correct reading orientation
    if needs_180_rotation:
        working_img = cv2.rotate(working_img, cv2.ROTATE_180)
        
    return working_img

def deskew_and_clean(warped_color, manual_rotation=None):
    """
    Corregge l'orientamento e migliora la qualità dell'immagine
    
    Args:
        warped_color (numpy.ndarray): Immagine di input
        manual_rotation (int, optional): Rotazione manuale (0=nessuna, 1=90°, 2=180°, 3=270°)
    
    Returns:
        dict: Dizionario contenente le varie versioni elaborate dell'immagine
    """
    # First, correct the orientation
    corrected = correct_orientation(warped_color, manual_rotation)
    
    # convert to grayscale
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    # CLAHE (contrast limited adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # Denoise
    den = cv2.fastNlMeansDenoising(cl, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold (for barcode/text clarity)
    thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 10)

    # Morphological operations to close small holes (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Sharpening (unsharp mask)
    gaussian = cv2.GaussianBlur(den, (0,0), sigmaX=3)
    unsharp = cv2.addWeighted(den, 1.5, gaussian, -0.5, 0)

    return {
        "corrected": corrected,
        "gray": gray,
        "clahe": cl,
        "denoised": den,
        "threshold": thr,
        "morph": morph,
        "unsharp": unsharp
    }

def main(input_path, output_prefix="out", manual_rotation=None, show_result=False):
    """
    Preelabora un'immagine di carta d'identità
    
    Args:
        input_path (str): Percorso dell'immagine da elaborare
        output_prefix (str): Prefisso per i file di output
        manual_rotation (int, optional): Rotazione manuale (0=nessuna, 1=90°, 2=180°, 3=270°)
        show_result (bool): Se True, mostra il risultato in una finestra
    """
    img = cv2.imread(input_path)
    if img is None:
        print("Errore: impossibile aprire l'immagine:", input_path)
        return
    
    # Resize for faster processing
    img = resize(img, width=1200)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect document contour and warp
    doc_cnt = detect_document_contour(gray)
    if doc_cnt is not None:
        warped = four_point_transform(orig, doc_cnt * (orig.shape[0]/img.shape[0]) if False else doc_cnt)
        # note: we used same scale so multiply isn't necessary here because resize kept ratios.
    else:
        # fallback: try to rotate/deskew using moments or Hough lines (simple fallback: use original)
        print("Avviso: nessun contorno documento rilevato, utilizzo immagine originale.")
        warped = orig.copy()

    # Process and correct orientation
    results = deskew_and_clean(warped, manual_rotation)

    # Save intermediate results for inspection
    cv2.imwrite(f"{output_prefix}_resized.jpg", img)
    cv2.imwrite(f"{output_prefix}_warped.jpg", warped)
    cv2.imwrite(f"{output_prefix}_corrected.jpg", results["corrected"])
    cv2.imwrite(f"{output_prefix}_clahe.jpg", results["clahe"])
    cv2.imwrite(f"{output_prefix}_denoised.jpg", results["denoised"])
    cv2.imwrite(f"{output_prefix}_threshold.jpg", results["threshold"])
    cv2.imwrite(f"{output_prefix}_morph.jpg", results["morph"])
    cv2.imwrite(f"{output_prefix}_unsharp.jpg", results["unsharp"])
    
    # Show results if requested
    if show_result:
        try:
            # Create a window with smaller size for easy viewing
            display_img = resize(results["corrected"], width=800)
            cv2.imshow("Documento Corretto", display_img)
            print("Premi un tasto qualsiasi per continuare...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Impossibile mostrare l'immagine. Funzione disponibile solo in ambiente desktop.")

    print("Elaborazione completata. File salvati con prefisso:", output_prefix)
    print("Suggerimento: controlla *_warped.jpg e *_threshold.jpg per la leggibilità del barcode.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilizzo: python preprocess.py input.jpg [output_prefix] [rotazione_manuale] [mostra]")
        print("  input.jpg: percorso dell'immagine da elaborare")
        print("  output_prefix: prefisso per i file di output (default: out)")
        print("  rotazione_manuale: (opzionale) 0=nessuna, 1=90°, 2=180°, 3=270°")
        print("  mostra: (opzionale) 1=mostra risultato in finestra")
        print("\nEsempi:")
        print("  python preprocess.py foto.jpg risultato")
        print("  python preprocess.py foto.jpg risultato 2")
        print("  python preprocess.py foto.jpg risultato 2 1")
    else:
        inp = sys.argv[1]
        pref = sys.argv[2] if len(sys.argv) > 2 else "out"
        rot = int(sys.argv[3]) if len(sys.argv) > 3 else None
        show = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
        
        if rot is not None:
            print(f"Applicazione rotazione manuale: {rot*90}°")
            
        main(inp, pref, rot, show)
