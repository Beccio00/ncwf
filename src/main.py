import sys
import subprocess
import os
from pyzbar.pyzbar import decode
import cv2

def main():
    if len(sys.argv) < 3:
        print("Uso: python src/main.py <input_image> <output_prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_prefix = sys.argv[2]

    # 1️⃣ Richiama preprocess.py
    print("[INFO] Preprocessing immagine...")
    subprocess.run([
        "python", "src/preprocess.py",
        input_path, output_prefix
    ])

    # 2️⃣ Carica l’immagine finale per pyzbar
    processed_img = f"{output_prefix}_threshold.jpg"
    if not os.path.exists(processed_img):
        print(f"[ERRORE] File {processed_img} non trovato!")
        sys.exit(1)

    img = cv2.imread(processed_img)

    # 3️⃣ Decodifica con pyzbar
    print("[INFO] Decodifica codice a barre...")
    decoded_objects = decode(img)

    if not decoded_objects:
        print("[ERRORE] Nessun codice trovato 😕")
    else:
        for obj in decoded_objects:
            print(f"[SUCCESSO] Tipo: {obj.type}")
            print(f"[SUCCESSO] Dati: {obj.data.decode('utf-8', errors='ignore')}")

if __name__ == "__main__":
    main()
