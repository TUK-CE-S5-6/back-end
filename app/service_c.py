import os
import time
import openai
import logging
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from elevenlabs.client import ElevenLabs

# 기본 URL 및 포트 설정 (필요에 따라 변경)
BASE_HOST = "http://localhost"
PORT = "8002"

app = FastAPI()

USER_FILES_FOLDER = "user_files"

# 정적 파일 제공 (extracted_audio 폴더)
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")
app.mount("/user_files", StaticFiles(directory="extracted_audio"), name="audio")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
SOUND_EFFECTS_FOLDER_ROOT = "user_files"

# OpenAI API 설정 (번역용)
OPENAI_API_KEY = "gpt-key"
openai.api_key = OPENAI_API_KEY

# ElevenLabs API 키 및 클라이언트 초기화
ELEVENLABS_API_KEY = "eleven-key"
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# 사운드 이펙트 파일 저장 폴더
SOUND_EFFECTS_FOLDER = os.path.join("extracted_audio", "sound_effects")
os.makedirs(SOUND_EFFECTS_FOLDER, exist_ok=True)

def translate_text(text: str) -> str:
    """
    OpenAI GPT API를 사용해 텍스트를 영어로 번역하는 함수.
    """
    try:
        logging.info(f"번역 전 텍스트: {text}")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Translate the following text to english."},
                {"role": "user", "content": text}
            ]
        )
        translated_text = response["choices"][0]["message"]["content"].strip()
        logging.info(f"번역 후 텍스트: {translated_text}")
        return translated_text
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        raise Exception(f"Translation failed: {e}")

def generate_sound_effect(text: str, output_path: str):
    logging.info("Generating sound effects...")
    result = elevenlabs.text_to_sound_effects.convert(
        text=text,
        duration_seconds=10,      # 원하는 사운드 이펙트 길이 (초)
        prompt_influence=0.3,      # 프롬프트 영향 (기본값 0.3)
    )
    with open(output_path, "wb") as f:
        for chunk in result:
            f.write(chunk)
    logging.info(f"Audio saved to {output_path}")
    return output_path

@app.post("/generate-sound-effect")
async def create_sound_effect(
    text: str = Form(...),
    user_id: int = Form(...)
):
    """
    user_id와 text를 Form 데이터로 받아,
    user_files/{user_id}/{파일명}으로 효과음을 생성 후 URL 반환
    """
    try:
        # 사용자별 폴더 생성
        user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        # 번역
        translated = translate_text(text)

        # 파일명 및 경로
        filename = f"{text}.mp3"
        output_path = os.path.join(user_folder, filename)

        # 사운드 이펙트 생성
        result = elevenlabs.text_to_sound_effects.convert(
            text=translated
        )
        with open(output_path, 'wb') as f:
            for chunk in result:
                f.write(chunk)

        # 결과 URL
        file_url = f"/user_files/{user_id}/{filename}"
        return JSONResponse({
            "message": "Sound effect 생성 완료",
            "file_url": file_url,
            "original_text": text,
            "translated_text": translated
        })

    except Exception as e:
        logging.error(f"generate-sound-effect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))