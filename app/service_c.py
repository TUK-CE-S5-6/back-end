import os
import logging
import psycopg2
import openai
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from elevenlabs.client import ElevenLabs
from datetime import datetime
from pydub import AudioSegment

# ----------------------------
# DB & API 설정
# ----------------------------
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

OPENAI_API_KEY = "gpt-key"
openai.api_key = OPENAI_API_KEY

ELEVENLABS_API_KEY = "eleven-key"
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ----------------------------
# FastAPI 기본 설정
# ----------------------------
app = FastAPI()
USER_FILES_FOLDER = "user_files"
THUMBNAIL_FOLDER = "thumbnails"

app.mount("/user_files", StaticFiles(directory=USER_FILES_FOLDER), name="user_files")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_FOLDER), name="thumbnails")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)

# ----------------------------
# 번역 함수
# ----------------------------
def translate_text(text: str) -> str:
    try:
        logging.info(f"번역 전: {text}")
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Translate the following text to English."},
                {"role": "user", "content": text}
            ]
        )
        translated = resp["choices"][0]["message"]["content"].strip()
        logging.info(f"번역 후: {translated}")
        return translated
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        raise

# ----------------------------
# 효과음 생성 API
# ----------------------------
@app.post("/generate-sound-effect")
async def create_sound_effect(
    text: str = Form(...),
    user_id: int = Form(...),          # ← 로그인 정보 직접 전달받음
    preset_name: str = Form(None),
    thumbnail_url: str = Form(None),
    duration_seconds: int = Form(10),
    prompt_influence: float = Form(0.3)
):
    try:
        # 사용자 폴더 생성
        user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        # 번역
        translated = translate_text(text)

        # 파일명 원문 그대로
        filename = f"{text}.mp3"
        output_path = os.path.join(user_folder, filename)

        # ElevenLabs 효과음 생성
        stream = elevenlabs.text_to_sound_effects.convert(
            text=translated,
            duration_seconds=duration_seconds,
            prompt_influence=prompt_influence
        )
        with open(output_path, "wb") as f:
            for chunk in stream:
                f.write(chunk)

        # 오디오 길이
        try:
            seg = AudioSegment.from_file(output_path)
            audio_duration = round(len(seg) / 1000.0, 3)
        except:
            audio_duration = float(duration_seconds)

        # URL
        file_url = f"/user_files/{user_id}/{filename}"

        # 프리셋 저장
        preset_id = None
        if preset_name:
            try:
                conn = get_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sound_effects (user_id, name, description, file_path, thumbnail_url, duration)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        user_id,
                        preset_name,
                        text,  # description에 원문 저장
                        file_url,
                        thumbnail_url,
                        audio_duration
                    ))
                    preset_id = cur.fetchone()[0]
                    conn.commit()
                conn.close()
            except Exception as db_err:
                logging.error(f"DB 저장 실패: {db_err}")

        return JSONResponse({
            "message": "Sound effect 생성 완료",
            "file_url": file_url,
            "original_text": text,
            "translated_text": translated,
            "audio_duration": audio_duration,
            "preset": {
                "id": preset_id,
                "name": preset_name,
                "thumbnail_url": thumbnail_url
            }
        })

    except Exception as e:
        logging.error(f"generate-sound-effect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
