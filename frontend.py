import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLO
model_proyecto = YOLO("best_model_sa_1.pt")  
model_proyecto_aug = YOLO("best_proyecto_conaumentacion.pt")

# Rango de detecciones por score
detections_by_score_range = {f"{i * 10}%-{(i + 1) * 10}%": 0 for i in range(10)}
detections_by_score_range_aug = {f"{i * 10}%-{(i + 1) * 10}%": 0 for i in range(10)}

# Título de la aplicación
st.title("Reconocimiento de imágenes de plantas de papa con YOLO")
st.text("Seleccione una imagen y presione el botón de 'Procesar' para detectar las plantas de papa")

# Slider para manejar la confianza
score_threshold = st.slider("Seleccione el score mínimo de confianza para detección", 0.0, 1.0, 0.3, 0.1)

# Carga de la imagen
uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"])

def draw_detections(image, detections, names, score_threshold, score_ranges):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    class_counts = {}

    for detection in detections:
        x0, y0, x1, y1 = map(int, detection[:4])
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = names[cls]
        
        score_range_key = f"{int(score * 10) * 10}%-{int(score * 10 + 1) * 10}%"
        score_ranges[score_range_key] += 1

        if score >= score_threshold:
            class_counts[object_name] = class_counts.get(object_name, 0) + 1
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0, y0 - 10), f"{object_name}: {score}", fill="red", font=font)
    
    return class_counts, score_ranges

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2, col3 = st.columns([2, 4, 4])
    
    with col1:
        st.image(image, caption="Imagen cargada", use_container_width=True)
    
    if st.button("Procesar"):
        progress_bar = st.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        image_v11 = image.copy()
        image_v11_aug = image.copy()
        
        result_v11 = model_proyecto.predict(image, max_det=1500)
        result_v11_aug = model_proyecto_aug.predict(image, max_det=1500)
        
        class_counts_v11, detections_by_score_range = draw_detections(
            image_v11, result_v11[0].boxes.data, model_proyecto.names, score_threshold, detections_by_score_range
        )
        
        class_counts_v11_aug, detections_by_score_range_aug = draw_detections(
            image_v11_aug, result_v11_aug[0].boxes.data, model_proyecto_aug.names, score_threshold, detections_by_score_range_aug
        )
        
        with col2:
            st.image(image_v11, caption="Imagen con detecciones sin aumentación", use_container_width=True)
            st.table(pd.DataFrame(class_counts_v11.items(), columns=['Clase', 'Ocurrencias']))
            st.subheader('Total por rango de score')
            st.table(pd.DataFrame(detections_by_score_range.items(), columns=['Rango de Score', 'Total Detecciones']))
        
        with col3:
            st.image(image_v11_aug, caption="Imagen con detecciones con aumentación", use_container_width=True)
            st.table(pd.DataFrame(class_counts_v11_aug.items(), columns=['Clase', 'Ocurrencias']))
            st.subheader('Total por rango de score')
            st.table(pd.DataFrame(detections_by_score_range_aug.items(), columns=['Rango de Score', 'Total Detecciones']))
