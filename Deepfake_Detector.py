# deepfake_streamlit_overlay_v6.py
import os
import warnings
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import pandas as pd
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt

# --------------------------
# Config et suppression logs
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

MODEL_PATH = "test.h5"
IMG_SIZE = (128, 128)

# --------------------------
# Charger le modèle
# --------------------------
@st.cache_resource
def load_model_cached():
    return keras.models.load_model(MODEL_PATH)

model = load_model_cached()

# --------------------------
# Prédiction sur image
# --------------------------
def predict_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Deepfake" if pred > 0.5 else "Réel"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, float(confidence)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🛡️ Deepfake Detector - Vidéo avec Tableau et Statistiques")
st.write("Analyse images et vidéos avec overlay, graphique, tableau par frame, résumé statistique et export vidéo.")

upload_type = st.radio("Type de fichier :", ["Image", "Vidéo"])
uploaded_file = st.file_uploader("Choisissez un fichier", type=["jpg","png","mp4","avi","mov"])

if upload_type == "Vidéo" and uploaded_file:
    overlay_alpha = st.slider("Transparence overlay", 0.0, 1.0, 0.2, 0.05)
    max_frames = st.slider("Nombre max de frames à analyser", 50, 500, 200, 10)

if uploaded_file:
    if upload_type == "Image":
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label, confidence = predict_image(image)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption=f"Label: {label} | Probabilité: {confidence:.2%}",
                 use_container_width=True)
        st.progress(confidence)

    else:  # Vidéo
        temp_video_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        temp_output = NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (frame_width, frame_height))

        frame_count = 0
        total_conf = 0
        stframe = st.empty()
        chart_data = pd.DataFrame({"Probabilité": []})
        chart = st.line_chart(chart_data)
        table_df = pd.DataFrame(columns=["Frame", "Label", "Probabilité"])
        frame_labels = {"Réel":0, "Deepfake":0}

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            label, confidence = predict_image(frame)
            total_conf += confidence
            frame_count += 1
            frame_labels[label] += 1

            # Overlay couleur
            overlay = frame.copy()
            color = (0, 255, 0) if label == "Réel" else (0, 0, 255)
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), color, -1)
            cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0, frame)

            # Texte
            cv2.putText(frame, f"{label} | {confidence:.2%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Écriture frame
            out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            chart.add_rows(pd.DataFrame({"Probabilité": [confidence]}))

            table_df = pd.concat([table_df, pd.DataFrame({
                "Frame": [frame_count],
                "Label": [label],
                "Probabilité": [f"{confidence:.2%}"]
            })], ignore_index=True)

        # Résultat final
        if frame_count > 0:
            avg_conf = total_conf / frame_count
            final_label = "Deepfake" if avg_conf > 0.5 else "Réel"
            st.success(f"✅ Résultat final vidéo : {final_label} | Probabilité moyenne : {avg_conf:.2%}")

            # Tableau interactif
            st.subheader("📊 Détail par frame")
            st.dataframe(table_df)

            # Résumé statistique
            st.subheader("📈 Résumé statistique")
            st.write(f"Nombre de frames analysées : {frame_count}")
            st.write(f"Réel : {frame_labels['Réel']} frames ({frame_labels['Réel']/frame_count:.2%})")
            st.write(f"Deepfake : {frame_labels['Deepfake']} frames ({frame_labels['Deepfake']/frame_count:.2%})")

            # Graphique camembert
            fig, ax = plt.subplots()
            ax.pie([frame_labels['Réel'], frame_labels['Deepfake']],
                   labels=["Réel", "Deepfake"],
                   colors=["green","red"],
                   autopct="%1.1f%%",
                   startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Bouton téléchargement vidéo
            with open(temp_output.name, "rb") as f:
                st.download_button("📥 Télécharger la vidéo analysée",
                                   data=f,
                                   file_name=f"analyzed_{uploaded_file.name}",
                                   mime="video/mp4")

        cap.release()
        out.release()
        os.remove(temp_video_path)
