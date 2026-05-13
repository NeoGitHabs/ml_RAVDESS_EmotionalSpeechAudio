from fastapi import FastAPI, HTTPException, UploadFile, File
from torchaudio import transforms as T
import torch.nn.functional as F
import streamlit as st
import torch.nn as nn
import soundfile as sf
import torchaudio
import tempfile
import uvicorn
import torch
import io
import os


class CheckAudioEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.second(self.first(x))


emotion_map = {
    0: "neutral", 1: "calm",    2: "happy",   3: "sad",
    4: "angry",   5: "fearful", 6: "disgust",  7: "surprised"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CheckAudioEmotion()
# vocab = torch.load("vocab_RAVDESS_EmotionalSpeechAudio.pth", map_location=device, weights_only=False)
model.load_state_dict(torch.load('model_RAVDESS_EmotionalSpeechAudio.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = nn.Sequential(
    T.MelSpectrogram(sample_rate=48000, n_mels=128, n_fft=1024, hop_length=512),
    T.AmplitudeToDB()
)

max_len = 500


def preprocess(waveform, sr):
    # Стерео → моно
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.T  # (samples, ch) → (ch, samples)
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ресемплинг
    if sr != 48000:
        waveform = T.Resample(orig_freq=sr, new_freq=48000)(waveform)

    # Спектрограмма
    spec = transform(waveform).squeeze(0)

    # Паддинг / обрезка
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    else:
        spec = F.pad(spec, (0, max_len - spec.shape[1]))

    return spec


app = FastAPI()


@app.post('/predict')
async def predict_emotion(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Пустой файл')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf)

        spec = preprocess(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_idx = torch.argmax(y_pred, dim=1).item()
            pred_emotion = emotion_map[pred_idx]

        return {'index': pred_idx, 'emotion': pred_emotion}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



# st.title('Audio Emotion Classifier')
# st.text('Загрузите аудио, и модель определит эмоцию.')
#
# uploaded = st.file_uploader('Выберите аудио', type=['wav', 'mp3', 'flac', 'ogg'])
#
# if not uploaded:
#     st.info('Загрузите аудио')
# else:
#     st.audio(uploaded)
#
#     if st.button('Распознать'):
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
#                 tmp.write(uploaded.read())
#                 tmp_path = tmp.name
#
#             wf, sr = sf.read(tmp_path, dtype='float32')
#             wf = torch.tensor(wf)
#             os.unlink(tmp_path)
#
#             spec = preprocess(wf, sr).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 y_pred = model(spec)
#                 pred_idx = torch.argmax(y_pred, dim=1).item()
#                 pred_emotion = emotion_map[pred_idx]
#
#             st.success(f'Модель думает, что это: {pred_emotion}')
#
#         except Exception as e:
#             st.error(f'Ошибка: {str(e)}')
