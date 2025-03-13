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
BASE_HOST = "localhost"
PORT = "8002"

app = FastAPI()

# 정적 파일 제공 (extracted_audio 폴더)
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")

# CORS 설정
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

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
async def create_sound_effect(text: str = Form(...)):
    try:
        # 입력 텍스트를 번역
        translated_text = translate_text(text)
        logging.debug(f"Translated text: {translated_text}")

        timestamp = int(time.time())
        output_filename = f"sound_effect_{timestamp}.mp3"
        output_path = os.path.join(SOUND_EFFECTS_FOLDER, output_filename)
        generate_sound_effect(translated_text, output_path)
        
        # 전체 URL 구성 (필요시 BASE_HOST와 PORT 수정)
        file_url = f"{BASE_HOST}:{PORT}/extracted_audio/sound_effects/{output_filename}"
        logging.info(f"Sound effect generated: {file_url}")
        return JSONResponse(
            content={
                "message": "Sound effect generated successfully",
                "file_url": file_url,
                "original_text": text,
                "translated_text": translated_text
            },
            status_code=200
        )
    except Exception as e:
        logging.error(f"Sound effect generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
