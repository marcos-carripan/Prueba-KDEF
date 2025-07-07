import numpy as np

labels_dict = {
    0: 'miedo',
    1: 'ira',
    2: 'disgusto',
    3: 'felicidad',
    4: 'neutro',
    5: 'tristeza',
    6: 'sorpresa'
}

np.save('modelos/labels_vggface.npy', labels_dict)
print("✅ Diccionario de etiquetas guardado correctamente.")
