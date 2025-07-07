import os

# Ruta raíz del dataset
base_path = os.path.join('data', 'KDEF')

# Verificar existencia de carpetas como AF01, AM02, etc.
subdirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
print(f"Se encontraron {len(subdirs)} carpetas dentro de {base_path}:\n{subdirs[:5]}...")

# Contar archivos .JPG dentro de las subcarpetas
jpg_total = 0
ejemplo_archivos = []

for subdir in subdirs:
    subdir_path = os.path.join(base_path, subdir)
    archivos = [f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')]
    jpg_total += len(archivos)
    if len(ejemplo_archivos) < 1 and archivos:
        ejemplo_archivos = archivos[:5]

print(f"\nTotal de imágenes .JPG encontradas: {jpg_total}")
print(f"Ejemplo de archivos JPG encontrados: {ejemplo_archivos}")
