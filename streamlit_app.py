import cv2
import streamlit as st
from PIL import Image
import time
import pandas as pd
from io import BytesIO
import ultralytics 
from ultralytics import YOLO
import numpy as np

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
        #st.text(object_names)
        
        # Simulación de proceso con la barra de progreso
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # Procesar la imagen con el modelo YOLO
        original_image=np.array(image)
        result = model(image)  # Usar predict para evaluar la imagen    
        result_predict=model.predict(image)
        class_counts = {}
        for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name =  model.names[cls]                    
                    # Dibuja la caja en la imagen y tambien la probabilidad
                    if(score>0.65):
                        # Contar las ocurrencias de cada clase para mostrarlas al final
                        if object_name in class_counts:
                            class_counts[object_name] += 1
                        else:
                            class_counts[object_name] = 1
                        cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 255, 0), 2)                    
                        # Añade la etiqueta y la probabilidad a la caja
                        label = f'{object_name}: {score}'
                        cv2.putText(original_image, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)                                          
                    
        st.image(original_image, caption="Imagen con detecciones", use_column_width=True)
        st.subheader('Conteo de clases detectadas')

        class_count_df = pd.DataFrame(class_counts.items(), columns=['Clase', 'Ocurrencias'])
        st.table(class_count_df)


