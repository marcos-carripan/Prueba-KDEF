# preprocesar_dataset.py
import os
import cv2
import dlib
import numpy as np

# Directorios
DATASET_DIR = 'data/KDEF'
OUTPUT_DIR = 'data/preprocesado'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detector de caras de dlib
detector = dlib.get_frontal_face_detector()

# Diccionario de emociones
emociones_dict = {
    'AF': 0,  # miedo
    'AN': 1,  # ira
    'DI': 2,  # disgusto
    'HA': 3,  # felicidad
    'NE': 4,  # neutro
    'SA': 5,  # tristeza
    'SU': 6   # sorpresa
}

IMG_SIZE = 224  # requerido por VGG-Face

X = []
y = []

for carpeta in os.listdir(DATASET_DIR):
    carpeta_path = os.path.join(DATASET_DIR, carpeta)
    if not os.path.isdir(carpeta_path):
        continue

    for archivo in os.listdir(carpeta_path):
        if archivo.endswith('.JPG'):
            ruta = os.path.join(carpeta_path, archivo)
            img = cv2.imread(ruta)
            if img is None:
                continue

            # Convertir a RGB (desde BGR de OpenCV)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rostros = detector(gris)

            if len(rostros) == 0:
                continue

            x1 = max(rostros[0].left(), 0)
            y1 = max(rostros[0].top(), 0)
            x2 = min(rostros[0].right(), rgb.shape[1])
            y2 = min(rostros[0].bottom(), rgb.shape[0])
            rostro = rgb[y1:y2, x1:x2]

            if rostro.size == 0:
                continue

            try:
                rostro = cv2.resize(rostro, (IMG_SIZE, IMG_SIZE))
            except cv2.error:
                continue

            codigo_emocion = archivo[4:6]
            etiqueta = emociones_dict.get(codigo_emocion, None)

            if etiqueta is not None:
                X.append(rostro)
                y.append(etiqueta)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUTPUT_DIR, 'imagenes.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'etiquetas.npy'), y)

print(f"✅ Preprocesamiento completado: {len(X)} imágenes procesadas.")

