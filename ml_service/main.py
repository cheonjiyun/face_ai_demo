from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
import numpy as np
from pydantic import BaseModel
from sklearn.externals import joblib
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 모델 로드
scaler = joblib.load("scaler.pkl")
svm = joblib.load("svm_model.pkl")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return JSONResponse({"error": "얼굴 감지 실패"}, status_code=400)

    # 벡터 추출
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape
    coords = []
    for lm in landmarks:
        coords.extend([lm.x * w, lm.y * h])

    X_scaled = scaler.transform([coords])
    proba = svm.predict_proba(X_scaled)[0]
    labels = svm.classes_

    label_proba = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)

    # GPT 생성
    prompt = "이 얼굴의 관상 확률:\n\n"
    for label, p in label_proba:
        prompt += f"- {label}: {p*100:.1f}%\n"
    prompt += "이걸 종합해서 재미있는 해석 문구를 만들어줘."

    gpt_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 관상 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    generated = gpt_resp.choices[0].message.content.strip()

    return {"proba": label_proba, "gpt": generated}
