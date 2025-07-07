import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Cargar datos preprocesados
X = np.load('data/preprocesado/imagenes.npy')
y = np.load('data/preprocesado/etiquetas.npy')

# Normalizar y dar formato
X = X.astype('float32') / 255.0
X = np.expand_dims(X, axis=-1)  # (n, 64, 64, 1)

# Etiquetas a números
emociones = sorted(set(y))
emocion_a_id = {emocion: idx for idx, emocion in enumerate(emociones)}
y_numerico = np.array([emocion_a_id[e] for e in y])
y_cat = to_categorical(y_numerico, num_classes=len(emociones))

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# Modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(emociones), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=2)

# Evaluación
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión del modelo en conjunto de prueba: {acc*100:.2f}%")

# Mapa etiquetas
print("\nMapa de etiquetas:")
for emocion, idx in emocion_a_id.items():
    print(f"{idx}: {emocion}")
