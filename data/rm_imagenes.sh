#!/bin/bash

# Define las rutas de las carpetas
carpeta_imagenes="./cropped_images"
carpeta_etiquetas="./cropped_labels"

# Recorre cada archivo de imagen en la carpeta de imágenes
for imagen in "$carpeta_imagenes"/*.jpg; do
    # Extrae el nombre del archivo sin extensión
    nombre_archivo=$(basename "$imagen" .jpg)
    
    # Verifica si existe el archivo de etiqueta correspondiente
    if [[ ! -f "$carpeta_etiquetas/$nombre_archivo.txt" ]]; then
        # Si no existe el archivo de etiqueta, elimina la imagen
        echo "Eliminando $imagen, no tiene etiqueta correspondiente"
        rm "$imagen"
    fi
done

