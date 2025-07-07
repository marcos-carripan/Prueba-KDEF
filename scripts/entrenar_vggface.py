import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.image import resize

# Cargar dataset preprocesado
X = np.load('data/preprocesado/imagenes.npy')
y = np.load('data/preprocesado/etiquetas.npy')

# Ajustar canales
if X.ndim == 3:
    X = np.expand_dims(X, -1)
if X.shape[-1] != 3:
    X = np.repeat(X, 3, axis=-1)

# Redimensionar y normalizar
X = resize(X, [224, 224]).numpy()
X = preprocess_input(X.astype('float32'))

# Etiquetas categóricas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# División entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

print(f"✅ Imágenes: {X.shape}")
print(f"✅ Etiquetas categóricas: {y_cat.shape}")
print(f"✅ Clases: {le.classes_}")

# -------- Construcción manual del modelo VGG-Face --------
def build_vggface():
    input_layer = Input(shape=(224, 224, 3))
    
    # Bloque 1
    x = Conv2D(64, (3,3), padding='same', activation='relu', name='conv1_1')(input_layer)
    x = Conv2D(64, (3,3), padding='same', activation='relu', name='conv1_2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='pool1')(x)
    
    # Bloque 2
    x = Conv2D(128, (3,3), padding='same', activation='relu', name='conv2_1')(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu', name='conv2_2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='pool2')(x)
    
    # Bloque 3
    x = Conv2D(256, (3,3), padding='same', activation='relu', name='conv3_1')(x)
    x = Conv2D(256, (3,3), padding='same', activation='relu', name='conv3_2')(x)
    x = Conv2D(256, (3,3), padding='same', activation='relu', name='conv3_3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='pool3')(x)

    # Bloque 4
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_1')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_2')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='pool4')(x)

    # Bloque 5
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv5_1')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv5_2')(x)
    x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv5_3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='pool5')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    
    # Salida personalizada
    output = Dense(y_cat.shape[1], activation='softmax', name='classifier')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

# Crear el modelo base
model = build_vggface()

# Cargar los pesos VGG-Face
model.load_weights(os.path.expanduser('~/.deepface/weights/vgg_face_weights.h5'), by_name=True, skip_mismatch=True)

# Compilar y entrenar
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Precisión final en test: {acc * 100:.2f}%")

# Guardar modelo y etiquetas
os.makedirs('modelos', exist_ok=True)
model.save('modelos/modelo_vggface_entrenado.h5')
np.save('modelos/labels_vggface.npy', le.classes_)
