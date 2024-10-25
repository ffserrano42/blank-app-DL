import cv2
import streamlit as st
from PIL import Image
import time
import pandas as pd
from io import BytesIO
import ultralytics 
from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO("yolov8n.pt") 

# Título de la aplicación
st.title("Aplicación de reconocimiento de imágenes con YOLO")

# Carga de la imagen
uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"])

# Si se ha cargado una imagen, la mostramos y habilitamos el botón de "Procesar"
if uploaded_image:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Botón de "Procesar"
    if st.button("Procesar"):
        # Barra de progreso
        progress_bar = st.progress(0)
        st.text("procesando")
        object_names = list(model.names.values())
        st.text(object_names)
        
        # Simulación de proceso con la barra de progreso
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # Procesar la imagen con el modelo YOLO
        result = model(image)  # Usar predict para evaluar la imagen
        #st.text(result[0].boxes.data)
        for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name =  model.names[cls]
                    label = f'Clase detectada {object_name} probabilidad {score}'
                    st.text(f'{label} x0,y0 {x0},{y0} x1,y1 {x1},{y1}')


