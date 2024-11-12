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
model_proyecto_aug=YOLO("best_proyecto_v11_6.pt") # este es el modelo entrenado con datos aumentados
model_proyecto=YOLO("best_proyecto_v11.pt")  #este es el modelo entrenado sin datos aumentados.


# Título de la aplicación
st.title("Aplicación de reconocimiento de imágenes de plantas de papa con YOLO ")
st.text("seleccione una imagen y presione el botón de 'Procesar' para detectar las plantas de papa ")

#slider para manejar la confianza
score_threshold = st.slider("Seleccione el score mínimo de confianza para detección", 0.0, 1.0, 0.3, 0.1)

# Carga de la imagen
uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"])

# Si se ha cargado una imagen, la mostramos y habilitamos el botón de "Procesar"
if uploaded_image:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_image)
    #st.image(image, caption="Imagen cargada", use_column_width=True)
    col1,col2,col3=st.columns(3)
    with col1:
        st.image(image, caption="Imagen cargada", use_container_width=True)
    # Botón de "Procesar"
    if st.button("Procesar"):
        # Barra de progreso
        progress_bar = st.progress(0)
        st.text("procesando")
        #st.text(object_names)
        
        # Simulación de proceso con la barra de progreso
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)        
        # Procesar la imagen con el modelo YOLO
        original_image_v11=np.array(image)
        original_image_v11_aug=np.array(image)
        result_v11=model_proyecto.predict(image,max_det=1500)         
        result_v11_aug=model_proyecto_aug.predict(image,max_det=1500)           
        st.subheader('Conteo de clases detectadas con deteccion v11')        
        class_counts_v11 = {}
        for detection in result_v11[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name =  model_proyecto.names[cls]                    
                    # Dibuja la caja en la imagen y tambien la probabilidad
                    if(score>=score_threshold):
                        # Contar las ocurrencias de cada clase para mostrarlas al final
                        if object_name in class_counts_v11:
                            class_counts_v11[object_name] += 1
                        else:
                            class_counts_v11[object_name] = 1
                        cv2.rectangle(original_image_v11, (x0, y0), (x1, y1), (255, 0,0), 1)                    
                        # Añade la etiqueta y la probabilidad a la caja
                        label = f'{object_name}: {score}'
                        cv2.putText(original_image_v11, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 1)                                         
        print(class_counts_v11)
        with col2:
            st.image(original_image_v11, caption="Imagen con detecciones sin aumentacion", use_container_width=True)      
            class_count_df_11 = pd.DataFrame(class_counts_v11.items(), columns=['Clase', 'Ocurrencias'])
            st.table(class_count_df_11)
        class_counts_v11_aug = {}
        for detection in result_v11_aug[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name =  model_proyecto_aug.names[cls]                    
                    # Dibuja la caja en la imagen y tambien la probabilidad
                    if(score>=score_threshold):
                        # Contar las ocurrencias de cada clase para mostrarlas al final
                        if object_name in class_counts_v11_aug:
                            class_counts_v11_aug[object_name] += 1
                        else:
                            class_counts_v11_aug[object_name] = 1
                        cv2.rectangle(original_image_v11_aug, (x0, y0), (x1, y1), (255, 0,0), 1)                    
                        # Añade la etiqueta y la probabilidad a la caja
                        label = f'{object_name}: {score}'
                        cv2.putText(original_image_v11_aug, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 1)                                         
        print(class_counts_v11_aug)        
        with col3:
            st.image(original_image_v11_aug, caption="Imagen con detecciones con aumentacion", use_container_width=True)      
            class_count_df_11_aug = pd.DataFrame(class_counts_v11_aug.items(), columns=['Clase', 'Ocurrencias'])
            st.table(class_count_df_11_aug)



