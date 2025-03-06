import streamlit as st
import librosa
import numpy as np

# 🎵 Configuración de la página
st.title("Analizador de Beats 🎶")

# 📂 Subir el archivo de audio
audio_file = st.file_uploader("Sube tu beat aquí", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file, format="audio/mp3")
    
    # 📀 Analizar el beat
    with st.spinner("Analizando el audio..."):
        try:
            y, sr = librosa.load(audio_file, sr=None, duration=30)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_index = np.argmax(np.sum(chroma, axis=1)) if chroma.shape[1] > 0 else 0
            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key = keys[key_index]
            energy = np.mean(librosa.feature.rms(y=y))
            
            # 🔥 Clasificar género
            generos = {
                "Trap": (130, 160),
                "Reggaeton": (85, 100),
                "Drill": (140, 150),
                "Dancehall": (90, 110),
                "Afrobeats": (95, 120),
                "Hip-Hop": (80, 110),
                "R&B": (60, 90),
                "Dembow": (100, 130),
                "Moombahton": (100, 108),
                "Latin Pop": (95, 110)
            }
            genero_detectado = "Desconocido"
            for genero, (bpm_min, bpm_max) in generos.items():
                if bpm_min <= bpm <= bpm_max:
                    genero_detectado = genero
                    break
            
            # 🎤 Sugerir artista
            artistas = {
                "Trap": ["Drake", "Travis Scott", "Future", "Lil Baby", "21 Savage"],
                "Reggaeton": ["Bad Bunny", "J Balvin", "Feid", "Rauw Alejandro"],
                "Drill": ["Pop Smoke", "Fivio Foreign", "Eladio Carrión"],
                "Hip-Hop": ["Drake", "Kanye West", "J. Cole", "Kendrick Lamar"],
                "R&B": ["The Weeknd", "Chris Brown", "Brent Faiyaz"]
            }
            artista_sugerido = np.random.choice(artistas.get(genero_detectado, ["Artista Desconocido"]))
            
            # 📊 Mostrar resultados
            st.subheader("Resultados del Análisis:")
            st.write(f"**🎵 Género Detectado:** {genero_detectado}")
            st.write(f"**📊 BPM:** {bpm:.2f}")
            st.write(f"**🎼 Tono:** {key}")
            st.write(f"**⚡ Energía:** {energy:.4f}")
            st.write(f"**🎤 Artista Sugerido:** {artista_sugerido}")
        except Exception as e:
            st.error(f"Error al analizar el audio: {e}")
