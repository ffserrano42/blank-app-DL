import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLO
model_proyecto = YOLO("best_model_sa_1.pt")  
model_proyecto_aug = YOLO("best_proyecto_conaumentacion.pt")

st.text("seleccione una imagen y presione el botón de 'Procesar' para detectar las plantas de papa ")