import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import os
import librosa
import numpy as np
import threading
from tensorflow.keras.models import load_model

# Modelin yolu
MODEL_PATH = 'P2_EfeDemir_1904040026_Model.h5'

# Duygu etiketleri
EMOTIONS = ['Sinirli', 'Sakin', 'Mutlu', 'Üzgün']

# Ses eşik değeri
THRESHOLD = 0.20

# Boyutları uygun hale getirme
def preprocess_input(data):
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=-1)
    return data

# Callback fonksiyonu
def callback(in_data, frame_count, time_info, status):
    global frames
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)

def predict_emotion_from_audio():
    try:
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024,
                        stream_callback=callback)

        global frames
        frames = []

        # Modeli yükle
        model = load_model(MODEL_PATH)

        # Tkinter penceresini güncelle
        def update_label():
            # Sürekli olarak sesi dinleyip işle
            while True:
                # Eğer 1 saniyelik ses verisi toplandıysa
                if len(frames) >= 44100 // 1024:
                    # Kaydedilen ses verisini Numpy array'e dönüştür
                    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                    
                    # Audio verisini kaydedilen sesin özelliklerine dönüştür
                    audio_data = audio_data.astype(np.float32) / 32767.0  # Normalizasyon

                    # MFCC özelliklerini çıkar
                    mfcc = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=40).T

                    # Giriş verisini hazırla
                    input_data = preprocess_input(mfcc)

                    # Tahmin yap
                    prediction = model.predict(input_data)

                    # Tahmini ekrana yazdır
                    predicted_emotions = {emotion: f"{round(probability * 100, 2)}%" for emotion, probability in zip(EMOTIONS, prediction[0])}
                    result_label.config(text=f"Tahmin edilen duygular:\n{predicted_emotions}")

                    # Frames'i temizle
                    frames.clear()

        # Tkinter penceresini güncelleme işlemini başlat
        update_thread = threading.Thread(target=update_label)
        update_thread.daemon = True
        update_thread.start()

        root.mainloop()

        # Akışı kapat
        stream.stop_stream()
        stream.close()
        p.terminate()

    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
        print("Hata:", e)

# Tkinter uygulaması oluşturma
root = tk.Tk()
root.title("Duygu Tanıma Uygulaması")
root.geometry("800x600")
root.configure(bg="lightgray")

# Başlık etiketi
title_label = tk.Label(root, text="Duygu Tanıma Uygulaması", font=("Helvetica", 24), bg="lightgray")
title_label.pack(pady=20)

# Sonuçları gösterecek etiket
result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="lightgray")
result_label.pack()

# Duygu tahminini başlatan buton
start_button = tk.Button(root, text="Duygu Tahminini Başlat", command=predict_emotion_from_audio, font=("Helvetica", 16), bg="lightblue", fg="black")
start_button.pack(pady=20)

root.mainloop()
