import os
import numpy as np
import librosa
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import JsonResponse
import librosa.display
import matplotlib.pyplot as plt
from django.core.files.base import ContentFile
from django.conf import settings
import io
import base64
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


llm = ChatGroq(
    temperature = 0.2,
    groq_api_key="place_grok_Api_key_here :)",
    model_name = "llama-3.3-70b-versatile"
)
# llm.invoke("he")

MODEL_PATH = "crydetection\\babyCryClassifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)


LABELS = ["belly_pain", "burping", "hungry", "discomfort","tired"]
def index(request):
    return render(request,'index.html')


def preprocess_audio(file_path):
    """Preprocess the uploaded audio file before prediction"""
    y, sr = librosa.load(file_path, sr=16000)  
    y = librosa.util.fix_length(y, size=16000)  

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  
    mfccs = np.expand_dims(mfccs, axis=-1) 

    mfccs = tf.image.resize(mfccs, (128, 128))  

    mfccs = np.expand_dims(mfccs, axis=0) 
    return mfccs

# def upload_audio(request):
#     if request.method == 'POST' and request.FILES.get('audio_file'):
#         audio_file = request.FILES['audio_file']
#         file_path = default_storage.save('uploads' + audio_file.name, audio_file)

#         # Process the file and predict
        # audio_features = preprocess_audio(file_path)
        # predictions = model.predict(audio_features)
        # predicted_label = LABELS[np.argmax(predictions)]

#         return JsonResponse({'prediction': predicted_label})

#     return render(request, 'upload.html')


def upload_audio(request):
    if request.method == "POST" and request.FILES.get("audio_file"):
        audio_file = request.FILES["audio_file"]
        name = request.POST.get("name")

        file_path = default_storage.save(f"uploads/{audio_file.name}", ContentFile(audio_file.read()))
        
        y, sr = librosa.load(default_storage.path(file_path), sr=None)

        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis="time", y_axis="mel")

        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram of {name}")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()
        buffer.seek(0)
        spectrogram_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        audio_features = preprocess_audio(file_path)
        predictions = model.predict(audio_features)
        predicted_label = LABELS[np.argmax(predictions)]
        prediction = "Hunger"  # Placeholder value

        promptScrape = PromptTemplate.from_template(
        f"""
        ##You are an expert in child health and you are here to help parents give complete explanation baby is crying due to {predicted_label}
        now give parents complete details why this is happening and how to tackle it basis on {predicted_label} just be short and consise

        #FORMAT
        Our model which is developed by Hamza predicted that audio that you uploaded belongs to class {predicted_label} and baby is crying due to {predicted_label} and reason can be .......

        #INSTRUCTION
        write in paragraphs short and consise
        """
        )
        chainScrape = promptScrape | llm
        res = chainScrape.invoke(input={'page_data':predicted_label})
        print(res.content)
        

        return JsonResponse({
            "prediction": predicted_label,
            "explanation": res.content,
            "spectrogram": f"data:image/png;base64,{spectrogram_base64}"
        })

    return render(request, 'upload.html')



