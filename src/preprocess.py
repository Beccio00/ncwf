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
    """
    Rileva i contorni di un documento in un'immagine in scala di grigi
    Utilizza diverse tecniche per aumentare la probabilità di trovare il contorno corretto
    
    Args:
        img_gray (numpy.ndarray): Immagine in scala di grigi
        
    Returns:
        numpy.ndarray: Array di 4 punti che definiscono il contorno, o None se non trovato
    """
    h, w = img_gray.shape
    img_area = h * w
    
    # Prima tecnica: edge detection standard poi trova il contorno a 4 punti più grande
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)
    
    # Dilatazione per chiudere piccoli spazi nel bordo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # Trova tutti i contorni
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Filtra i contorni per area minima (almeno il 20% dell'area totale)
    min_area = img_area * 0.2
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    
    # Ordina i contorni per area (dal più grande al più piccolo)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    # Cerca contorni con 4 punti (rettangoli)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Se abbiamo trovato un contorno con 4 punti, è probabilmente il documento
        if len(approx) == 4:
            print(f"Documento rilevato: area {cv2.contourArea(c)/(img_area)*100:.1f}% dell'immagine")
            return approx.reshape(4, 2)
    
    # Se non troviamo un contorno con 4 punti, proviamo con un approccio più aggressivo
    # Tecnica alternativa: thresholding adattivo
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    dilated2 = cv2.dilate(thresh, kernel, iterations=2)
    
    cnts2 = cv2.findContours(dilated2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts2:
        area = cv2.contourArea(c)
        # Ignora contorni troppo piccoli
        if area < min_area:
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Se il contorno ha troppi punti, cerchiamo di ricavarne uno rettangolare
        if len(approx) > 4:
            # Ottieni il rettangolo che racchiude il contorno
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            print(f"Approssimato documento con rettangolo: area {area/(img_area)*100:.1f}% dell'immagine")
            return box
        elif len(approx) == 4:
            print(f"Documento rilevato (metodo alternativo): area {area/(img_area)*100:.1f}% dell'immagine")
            return approx.reshape(4, 2)
    
    # Se ancora non abbiamo trovato nulla, prendiamo un rettangolo che occupi la maggior parte dell'immagine
    # ma con un piccolo margine
    print("Nessun contorno documento rilevato, utilizzo area intera con margini")
    margin = int(min(h, w) * 0.05)  # 5% di margine
    return np.array([
        [margin, margin],                 # Top left
        [w - margin, margin],             # Top right
        [w - margin, h - margin],         # Bottom right
        [margin, h - margin]              # Bottom left
    ])

def detect_card_orientation(img):
    """
    Determina se la carta d'identità è in formato verticale o orizzontale.
    Ruota l'immagine solo se necessario per assicurarsi che sia in orizzontale 
    (con il lato lungo in orizzontale).
    
    Args:
        img (numpy.ndarray): Immagine di input
        
    Returns:
        tuple: (immagine lavorata, era in formato verticale)
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    h, w = gray.shape
    
    # Check if portrait or landscape and make sure it's in landscape
    # Le carte d'identità italiane sono rettangolari con il lato lungo orizzontale
    needs_90_rotation = False
    if h > w:  # Immagine in formato verticale/portrait
        needs_90_rotation = True
        print("Immagine rilevata in formato verticale, ruotando in orizzontale")
        working_img = cv2.rotate(img.copy(), cv2.ROTATE_90_CLOCKWISE)
    else:
        working_img = img.copy()
        print("Immagine già in formato orizzontale")
    
    return working_img, needs_90_rotation

def correct_orientation(img, manual_rotation=None):
    """
    Corregge l'orientamento di una carta d'identità per assicurarsi che sia 
    in formato orizzontale (lato lungo orizzontale).
    
    Procedimento:
    1. Se viene specificata una rotazione manuale, la applica direttamente
    2. Altrimenti:
       a. Verifica se la carta è in formato orizzontale (lato lungo orizzontale)
       b. Ruota solo se necessario per renderla orizzontale
    
    Parameters:
        img (numpy.ndarray): Immagine di input
        manual_rotation (int, optional): Override rilevamento automatico con rotazione manuale
                                        0 = nessuna rotazione
                                        1 = 90° in senso orario
                                        2 = 180°
                                        3 = 270° in senso orario (90° in senso antiorario)
    """
    # Se viene specificata una rotazione manuale, la applica direttamente
    if manual_rotation is not None:
        print(f"Utilizzo rotazione manuale: {manual_rotation * 90}°")
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
    
    # Rilevamento e correzione automatica dell'orientamento
    print("Analisi automatica dell'orientamento dell'immagine...")
    working_img, was_portrait = detect_card_orientation(img)
    
    # Riepilogo delle operazioni effettuate
    if was_portrait:
        print("Operazioni: Rotazione 90°")
    else:
        print("Operazioni: Nessuna rotazione necessaria")
        
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

def add_border(img, border_size=0):
    """
    Aggiunge un bordo all'immagine
    
    Args:
        img (numpy.ndarray): Immagine di input
        border_size (int): Dimensione del bordo in pixel
        
    Returns:
        numpy.ndarray: Immagine con bordo aggiunto
    """
    if border_size <= 0:
        return img
        
    return cv2.copyMakeBorder(
        img,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # Bordo bianco
    )

def main(input_path, output_prefix="out", manual_rotation=None, show_result=False, margin=10):
    """
    Preelabora un'immagine di carta d'identità
    
    Args:
        input_path (str): Percorso dell'immagine da elaborare
        output_prefix (str): Prefisso per i file di output
        manual_rotation (int, optional): Rotazione manuale (0=nessuna, 1=90°, 2=180°, 3=270°)
        show_result (bool): Se True, mostra il risultato in una finestra
        margin (int): Margine in pixel da aggiungere intorno al documento ritagliato
    """
    img = cv2.imread(input_path)
    if img is None:
        print("Errore: impossibile aprire l'immagine:", input_path)
        return
    
    # Resize for faster processing
    img = resize(img, width=1200)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect document contour - la funzione modificata ora restituisce sempre un contorno valido
    doc_cnt = detect_document_contour(gray)
    
    # Applica la trasformazione prospettica per ottenere una vista dall'alto del documento
    warped = four_point_transform(orig, doc_cnt)
    
    # Aggiungi un piccolo margine bianco per estetica, se richiesto
    if margin > 0:
        warped = add_border(warped, margin)

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
        print("Utilizzo: python preprocess.py input.jpg [output_prefix] [rotazione_manuale] [mostra] [margine]")
        print("  input.jpg: percorso dell'immagine da elaborare")
        print("  output_prefix: prefisso per i file di output (default: out)")
        print("  rotazione_manuale: (opzionale) 0=nessuna, 1=90°, 2=180°, 3=270°")
        print("  mostra: (opzionale) 1=mostra risultato in finestra")
        print("  margine: (opzionale) margine in pixel intorno al documento (default: 10)")
        print("\nEsempi:")
        print("  python preprocess.py foto.jpg risultato")
        print("  python preprocess.py foto.jpg risultato 2")
        print("  python preprocess.py foto.jpg risultato 2 1")
        print("  python preprocess.py foto.jpg risultato 2 1 20")
    else:
        inp = sys.argv[1]
        pref = sys.argv[2] if len(sys.argv) > 2 else "out"
        rot = int(sys.argv[3]) if len(sys.argv) > 3 else None
        show = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
        margin = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        
        if rot is not None:
            print(f"Applicazione rotazione manuale: {rot*90}°")
        
        print(f"Margine: {margin}px")
            
        main(inp, pref, rot, show, margin)
