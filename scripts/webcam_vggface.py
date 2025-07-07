# scripts/webcam_vggface.py

import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model

# Cargar modelo entrenado
print("🔄 Cargando clasificador entrenado...")
emotion_model = load_model("modelos/modelo_vggface_entrenado.h5")
print("✅ Modelo cargado correctamente.")

# Cargar etiquetas
label_dict = np.load("modelos/labels_vggface.npy", allow_pickle=True).item()

# Inicializar webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam iniciada...")

# Detector de rostros
detector = dlib.get_frontal_face_detector()

# Tamaño de entrada para VGG-Face
IMG_SIZE = 224

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises y RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Extraer rostro
        face_img = rgb[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        # Preprocesar
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_array = np.expand_dims(face_img, axis=0) / 255.0

        # Predicción
        prediction = emotion_model.predict(face_array, verbose=0)
        label_idx = np.argmax(prediction)
        label = label_dict[label_idx]
        confidence = prediction[0][label_idx]

        # Dibujar rectángulo y emoción
        text = f"Emocion: {label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Mostrar frame
    cv2.imshow("🎭 Reconocimiento de Emociones - VGG-Face", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
