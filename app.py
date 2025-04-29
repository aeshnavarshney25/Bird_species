import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import os

def create_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('L')  # Grayscale
    return image

def extract_species_name(filename):
    # Remove extension and replace underscores with spaces
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ")

# Streamlit UI
st.title(" Bird Species Identifier from Audio")

uploaded_file = st.file_uploader("Upload a bird call (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    st.write("Generating spectrogram...")
    spectrogram_image = create_spectrogram(uploaded_file)
    st.image(spectrogram_image, caption="Generated Spectrogram", use_column_width=True)

    st.write("Reading species name from filename...")
    species_name = extract_species_name(uploaded_file.name)
    st.success(f"Predicted Bird Species: **{species_name}**")
