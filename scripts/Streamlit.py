import os
import base64
import sys
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Insérer le chemin YOLOv9
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yolov9.detect import detect_in_streamlit  # Importer la nouvelle fonction

# Fonction pour convertir une image en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_str

# Chemin vers l'image locale
image_path = '../assets/images/background.png'


# Convertir l'image en base64
base64_image = get_base64_image(image_path)

# CSS personnalisé pour l'image de fond et la mise en page
background_image = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{base64_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

h2 {{
    font-size: 40px;
}}

div[data-baseweb="select"] > div {{
    margin-bottom: 15px;
    font-size: 18px;
}}

div[data-baseweb="slider"] > div {{
    margin-top: 10px;
    font-size: 18px;
}}

</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

# HTML personnalisé pour l'en-tête
st.markdown("""
    <div style="background-color: #2C3E50; padding: 10px; border-radius: 10px; margin-bottom: 35px;">
        <h2 style="color: #ECF0F1; text-align: center;"> Détection et Classification d'Objets en Mer avec YOLOv9 </h2>
    </div>
    """, unsafe_allow_html=True)

# Chemins vers les poids des modèles
weights_paths = [
    '../yolov9/weights/gelan-c-det.pt',
    '../yolov9/runs/train/Sea-Vessels-Dataset-2/weights/best_striped.pt'
]

# Chemins vers les vidéos
video_paths = [
    '../assets/videos/Videosample1.mp4',
]

# Extraction des noms des fichiers pour éviter d'afficher les chemins dans les menus déroulants
weights_names = [Path(weight).parent.parent.name for weight in weights_paths]
video_names = [Path(video).stem for video in video_paths]

# Menu déroulant : Sélection du modèle de poids
selected_weight_name = st.selectbox('Choisir le modèle de poids', weights_names)
selected_weights_path = weights_paths[weights_names.index(selected_weight_name)]

# Menu déroulant : Sélection de la vidéo
selected_video_name = st.selectbox('Choisir la vidéo', video_names)
selected_video_path = video_paths[video_names.index(selected_video_name)]

# Curseur pour le seuil de confiance
confidence_threshold = st.slider('Seuil de confiance', 0, 100, 25)

# Checkbox pour enregistrer les résultats
save_results = st.checkbox('Enregistrer les résultats', value=True)

# Variables de contrôle
stop_video = False

# Fonction pour traiter et afficher la vidéo
def process_video(video_path, weights_path, confidence_threshold, save_results):
    global stop_video
    # Créer affichage Streamlit pour la vidéo
    video_stream = st.empty()
    # Appeler la fonction detect_in_streamlit
    detect_in_streamlit(
        weights=weights_path,
        source=video_path,
        imgsz=(640, 640),
        conf_thres=confidence_threshold / 100.0,
        iou_thres=0.45,
        device='0',
        save_txt=False,
        save_conf=False,
        save_crop=False,
        view_img=True,
        video_stream=video_stream,
        save_results=save_results
    )

# Fonction pour charger les données de performance du modèle
def load_performance_data(model_path):
    csv_path = Path(model_path).parent.parent / 'results.csv'  # Localisation correcte du fichier CSV
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()  # Supprimer les espaces superflus dans les noms de colonnes
    return data

# Fonction pour tracer les graphiques de performance
def plot_performance(data):
    fig, ax = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle('Performance Metrics', fontsize=16)

    metrics = [
        ('train/box_loss', 'Train Box Loss'),
        ('val/box_loss', 'Validation Box Loss'),
        ('train/cls_loss', 'Train Class Loss'),
        ('val/cls_loss', 'Validation Class Loss'),
        ('train/dfl_loss', 'Train DFL Loss'),
        ('val/dfl_loss', 'Validation DFL Loss'),
        ('metrics/precision', 'Precision'),
        ('metrics/recall', 'Recall'),
        ('metrics/mAP_0.5', 'mAP@0.5'),
        ('metrics/mAP_0.5:0.95', 'mAP@0.5:0.95')
    ]

    for idx, (metric, title) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax[row, col].plot(data['epoch'], data[metric], label=title)
        ax[row, col].set_title(title)
        ax[row, col].set_xlabel('Epoch')
        ax[row, col].set_ylabel(metric.split('/')[1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)

# Checkbox pour afficher/masquer les graphiques
show_graphs = st.checkbox('Afficher les graphiques de performance')

# Lire les données de performance et les afficher si la checkbox est cochée
if show_graphs:
    performance_data = load_performance_data(selected_weights_path)
    plot_performance(performance_data)

# Disposition des boutons sur la même ligne
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])

with col2:
    start_button = st.button('Detect')

with col4:
    stop_button = st.button('Stop')

# Vérifier si les boutons ont été pressés
if start_button:
    stop_video = False
    process_video(selected_video_path, selected_weights_path, confidence_threshold, save_results)

if stop_button:
    stop_video = True

# Créer un conteneur pour la vidéo en pleine largeur
video_container = st.container()

with video_container:
    if 'video_stream' in locals():
        video_stream = st.empty()
