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
model_proyecto=YOLO("best_model_sa_1.pt")  #este es el modelo entrenado sin datos aumentados.
model_proyecto_aug=YOLO("best_proyecto_v11_6.pt") # este es el modelo entrenado con datos aumentados

#rango de detecciones por score para que en cada modelo se pueda mostrar las detecciones por score
detections_by_score_range = {
    "0-0.1": 0,
    "0.1-0.2": 0,
    "0.2-0.3": 0,
    "0.3-0.4": 0,
    "0.4-0.5": 0,
    "0.5-0.6": 0,
    "0.6-0.7": 0,
    "0.7-0.8": 0,
    "0.8-0.9": 0,
    "0.9-1": 0
}

#rango de deteccion para el modelo con aumentacion
detections_by_score_range_aug = {
    "0-0.1": 0,
    "0.1-0.2": 0,
    "0.2-0.3": 0,
    "0.3-0.4": 0,
    "0.4-0.5": 0,
    "0.5-0.6": 0,
    "0.6-0.7": 0,
    "0.7-0.8": 0,
    "0.8-0.9": 0,
    "0.9-1": 0
}

# Título de la aplicación
st.title("Reconocimiento de imágenes de plantas de papa con YOLO ")
st.text("seleccione una imagen y presione el botón de 'Procesar' para detectar las plantas de papa ")

#slider para manejar la confianza
score_threshold = st.slider("Seleccione el score mínimo de confianza para detección", 0.0, 1.0, 0.3, 0.1)
print(score_threshold)
# Carga de la imagen
uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png", "jpeg"])

# Si se ha cargado una imagen, la mostramos y habilitamos el botón de "Procesar"
if uploaded_image:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_image)
    #st.image(image, caption="Imagen cargada", use_column_width=True)
    col1,col2,col3=st.columns([2,4,4])
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
        result_v11=model_proyecto.predict(image,max_det=1500)#se coloca el maximo de detecciones posibles         
        result_v11_aug=model_proyecto_aug.predict(image,max_det=1500)#se coloca el maximo de detecciones posibles                   
        #st.subheader('Conteo de clases detectadas con deteccion v11')        
        class_counts_v11 = {}
        for detection in result_v11[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name =  model_proyecto.names[cls]                                                        
                    # Agrupa las detecciones por rango de score
                    if score < 0.1:
                        detections_by_score_range["0-0.1"] += 1
                    elif score < 0.2:
                        detections_by_score_range["0.1-0.2"] += 1
                    elif score < 0.3:
                        detections_by_score_range["0.2-0.3"] += 1
                    elif score < 0.4:
                        detections_by_score_range["0.3-0.4"] += 1
                    elif score < 0.5:
                        detections_by_score_range["0.4-0.5"] += 1
                    elif score < 0.6:
                        detections_by_score_range["0.5-0.6"] += 1
                    elif score < 0.7:
                        detections_by_score_range["0.6-0.7"] += 1
                    elif score < 0.8:
                        detections_by_score_range["0.7-0.8"] += 1
                    elif score < 0.9:
                        detections_by_score_range["0.8-0.9"] += 1
                    else:
                        detections_by_score_range["0.9-1"] += 1
                    # Dibuja la caja en la imagen y tambien la probabilidad
                    if(score>=score_threshold):                        
                        # Contar las ocurrencias de cada clase para mostrarlas al final
                        if object_name in class_counts_v11:
                            class_counts_v11[object_name] += 1
                            print(class_counts_v11[object_name])
                        else:
                            class_counts_v11[object_name] = 1
                        cv2.rectangle(original_image_v11, (x0, y0), (x1, y1), (255, 0,0), 1)                    
                        # Añade la etiqueta y la probabilidad a la caja
                        label = f'{object_name}: {score}'
                        cv2.putText(original_image_v11, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 1)                                               
        with col2:
            st.image(original_image_v11, caption="Imagen con detecciones sin aumentacion", use_container_width=True)      
            # Convertir el diccionario de conteo a un DataFrame
            class_count_df_11 = pd.DataFrame(class_counts_v11.items(), columns=['Clase', 'Ocurrencias'])          
            df_score_range = pd.DataFrame(list(detections_by_score_range.items()), columns=['Score Range', 'Total Detections'])
            st.table(class_count_df_11)
            st.subheader('total by score')
            st.table(df_score_range)

        # iniciamos el proceso para el modelo CON aumentacion    
        class_counts_v11_aug = {}
        for detection in result_v11_aug[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name =  model_proyecto_aug.names[cls]
                     # Agrupa las detecciones por rango de score
                    if score < 0.1:
                        detections_by_score_range_aug["0-0.1"] += 1
                    elif score < 0.2:
                        detections_by_score_range_aug["0.1-0.2"] += 1
                    elif score < 0.3:
                        detections_by_score_range_aug["0.2-0.3"] += 1
                    elif score < 0.4:
                        detections_by_score_range_aug["0.3-0.4"] += 1
                    elif score < 0.5:
                        detections_by_score_range_aug["0.4-0.5"] += 1
                    elif score < 0.6:
                        detections_by_score_range_aug["0.5-0.6"] += 1
                    elif score < 0.7:
                        detections_by_score_range_aug["0.6-0.7"] += 1
                    elif score < 0.8:
                        detections_by_score_range_aug["0.7-0.8"] += 1
                    elif score < 0.9:
                        detections_by_score_range_aug["0.8-0.9"] += 1
                    else:
                        detections_by_score_range_aug["0.9-1"] += 1                    
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
        with col3:
            st.image(original_image_v11_aug, caption="Imagen con detecciones con aumentacion", use_container_width=True)      
            class_count_df_11_aug = pd.DataFrame(class_counts_v11_aug.items(), columns=['Clase', 'Ocurrencias'])
            df_score_range_aug = pd.DataFrame(list(detections_by_score_range_aug.items()), columns=['Score Range', 'Total Detections'])
            st.table(class_count_df_11_aug)
            st.subheader('total by score')
            st.table(df_score_range_aug)
            



