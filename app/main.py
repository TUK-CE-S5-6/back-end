import os
import time
import openai
import psycopg2
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from whisper import load_model
from moviepy.editor import AudioFileClip, VideoFileClip
import logging
from elevenlabs import set_api_key, generate, Voice
from spleeter.separator import Separator
import shutil
from fastapi.staticfiles import StaticFiles

#########################
# PostgreSQL 설정
#########################
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

#########################
# FastAPI 앱 생성
#########################
app = FastAPI()

#########################
# CORS 설정
#########################
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#########################
# 정적 파일 제공 (비디오 및 오디오)
#########################
app.mount("/videos", StaticFiles(directory="uploaded_videos"), name="videos")
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")

#########################
# Pydantic 모델
#########################
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    user_id: int
    username: str

class TTSRequest(BaseModel):
    file_name: str

UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUDIO_FOLDER = "extracted_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

WHISPER_MODEL = "large"
model = load_model(WHISPER_MODEL)

#########################
# OpenAI API 설정
#########################
OPENAI_API_KEY = "open-ai-key"
openai.api_key = OPENAI_API_KEY

#########################
# ElevenLabs TTS 설정
#########################
ELEVENLABS_API_KEY = "eleven-key"
set_api_key(ELEVENLABS_API_KEY)

#########################
# PostgreSQL 연결 함수
#########################
def get_connection():
    """
    PostgreSQL에 연결하는 함수
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

#########################
# 루트 라우트
#########################
@app.get("/")
def read_root():
    return {"message": "Hello Root!"}

#########################
# 사용자 회원가입
#########################
@app.post("/users", response_model=User)
def create_user(data: UserCreate):
    """
    회원가입 기능 (username 중복 검사 포함)
    """
    conn = get_connection()
    curs = conn.cursor()

    # 중복 username 체크
    curs.execute("SELECT user_id FROM users WHERE username = %s;", (data.username,))
    row = curs.fetchone()
    if row:
        curs.close()
        conn.close()
        raise HTTPException(status_code=400, detail="이미 사용 중인 사용자 이름입니다.")

    # 사용자 삽입
    curs.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING user_id;", (data.username, data.password))
    new_id = curs.fetchone()[0]

    conn.commit()
    curs.close()
    conn.close()

    return {"user_id": new_id, "username": data.username}

#########################
# 사용자 목록 조회
#########################
@app.get("/users", response_model=list[User])
def list_users():
    """
    DB에 저장된 모든 사용자 목록 반환
    """
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id, username FROM users;")
    rows = curs.fetchall()
    curs.close()
    conn.close()

    return [{"user_id": r[0], "username": r[1]} for r in rows]

#########################
# 특정 사용자 조회
#########################
@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int):
    """
    user_id에 해당하는 사용자 정보 조회
    """
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id, username FROM users WHERE user_id = %s;", (user_id,))
    row = curs.fetchone()
    curs.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"user_id": row[0], "username": row[1]}

# 🎙️ Spleeter 실행 후, vocals.wav 및 accompaniment.wav의 실제 경로 탐색
def find_spleeter_output(base_folder: str, file_name: str):
    """
    Spleeter가 생성한 vocals.wav 및 accompaniment.wav의 실제 경로를 탐색하여 반환.
    """
    # 1️⃣ 기본적으로 `extracted_audio/{file_name}_audio/` 폴더를 탐색
    expected_folder = os.path.join(base_folder, f"{file_name}_audio")

    if not os.path.exists(expected_folder):
        raise FileNotFoundError(f"❌ {expected_folder} 경로가 존재하지 않습니다!")

    # 2️⃣ 해당 폴더 내부에서 `vocals.wav` 및 `accompaniment.wav`를 찾기
    for root, dirs, files in os.walk(expected_folder):
        if "vocals.wav" in files and "accompaniment.wav" in files:
            return os.path.join(root, "vocals.wav"), os.path.join(root, "accompaniment.wav")

    raise FileNotFoundError(f"❌ '{expected_folder}' 내부에 vocals.wav 또는 accompaniment.wav를 찾을 수 없습니다!")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_name = os.path.splitext(file.filename)[0]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # ✅ 기존 `extracted_audio/` 폴더 정리 (이전 처리물 삭제)
        extracted_audio_subfolder = os.path.join(AUDIO_FOLDER, f"{file_name}_audio")
        if os.path.exists(extracted_audio_subfolder):
            shutil.rmtree(extracted_audio_subfolder, ignore_errors=True)

        # ✅ 업로드된 파일 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # ✅ 🎥 비디오 길이(duration) 계산
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration

        # ✅ 🎼 비디오에서 오디오 추출
        extracted_audio_path = os.path.join(AUDIO_FOLDER, f"{file_name}_audio.mp3")
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(extracted_audio_path, codec='mp3')
        audio_clip.close()
        video_clip.close()

        # ✅ 🎙️ 음성과 배경음악 분리 (Spleeter 실행)
        try:
            separator = Separator("spleeter:2stems")
            separator.separate_to_file(extracted_audio_path, AUDIO_FOLDER)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Spleeter 실행 실패: {str(e)}")

        # ✅ Spleeter가 생성한 실제 폴더를 탐색하여 vocals.wav, accompaniment.wav 찾기
        vocals_path, bgm_path = find_spleeter_output(AUDIO_FOLDER, file_name)

        # ✅ 최종 경로로 이동 (폴더 구조 정리)
        fixed_vocals_path = os.path.join(AUDIO_FOLDER, f"{file_name}_vocals.wav")
        fixed_bgm_path = os.path.join(AUDIO_FOLDER, f"{file_name}_bgm.wav")

        shutil.move(vocals_path, fixed_vocals_path)
        shutil.move(bgm_path, fixed_bgm_path)

        # ✅ Spleeter가 만든 폴더 삭제
        shutil.rmtree(os.path.join(AUDIO_FOLDER, f"{file_name}_audio"), ignore_errors=True)

        # ✅ 📌 DB에 비디오 정보 저장
        conn = get_connection()
        curs = conn.cursor()

        curs.execute("""
            INSERT INTO videos (file_name, file_path, duration) 
            VALUES (%s, %s, %s) RETURNING video_id;
        """, (file.filename, file_path, duration))
        
        video_id = curs.fetchone()[0]

        # ✅ 📌 DB에 배경음(BGM) 저장
        curs.execute("""
            INSERT INTO background_music (video_id, file_path, volume) 
            VALUES (%s, %s, %s);
        """, (video_id, fixed_bgm_path, 1.0))  # 기본 볼륨 1.0

        conn.commit()
        curs.close()
        conn.close()

        # ✅ 📌 STT, 번역, TTS 순차 실행
        await transcribe_audio(fixed_vocals_path, video_id)
        await translate_video(video_id)
        await generate_tts(video_id)
        return await get_edit_data(video_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

#########################
# 📌 2. STT 변환 & 저장
#########################
async def transcribe_audio(audio_path: str, video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()

        # Whisper로 STT 실행
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"STT 변환 실패: {audio_path} 파일이 존재하지 않습니다.")

        result = model.transcribe(audio_path, word_timestamps=True)

        for segment in result["segments"]:
            start_time = float(segment["start"])
            end_time = float(segment["end"])

            curs.execute("""
                INSERT INTO transcripts (video_id, language, text, start_time, end_time)
                VALUES (%s, %s, %s, %s, %s);
            """, (video_id, "ko", segment["text"], start_time, end_time))

        conn.commit()
        curs.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")
    
#########################
# 📌 4. 번역 & 저장 (자동 실행)
#########################
async def translate_video(video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()

        # STT 데이터 가져오기
        curs.execute("SELECT transcript_id, text FROM transcripts WHERE video_id = %s;", (video_id,))
        transcripts = curs.fetchall()

        if not transcripts:
            raise HTTPException(status_code=404, detail="STT 데이터가 없습니다.")

        for transcript_id, text in transcripts:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "Translate the following Korean text into English."},
                          {"role": "user", "content": text}]
            )
            translated_text = response["choices"][0]["message"]["content"].strip()

            curs.execute("""
                INSERT INTO translations (transcript_id, language, text)
                VALUES (%s, %s, %s);
            """, (transcript_id, "en", translated_text))

        conn.commit()
        curs.close()
        conn.close()

        logging.info(f"번역 완료: video_id={video_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"번역 실패: {str(e)}")

#########################
# 📌 5. TTS 변환 & 저장 (자동 실행)
#########################
async def generate_tts(video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()

        # 번역된 데이터 가져오기 (번역 ID, 텍스트, 시작 시간)
        curs.execute("""
            SELECT t.translation_id, t.text, tr.start_time
            FROM translations t
            JOIN transcripts tr ON t.transcript_id = tr.transcript_id
            WHERE tr.video_id = %s;
        """, (video_id,))
        translations = curs.fetchall()

        if not translations:
            raise HTTPException(status_code=404, detail="번역된 데이터가 없습니다.")

        tts_output_dir = os.path.join(AUDIO_FOLDER, f"{video_id}_tts")
        os.makedirs(tts_output_dir, exist_ok=True)

        # ✅ 사용하고 싶은 voice_id 지정
        selected_voice_id = "5Af3x6nAIWjF6agOOtOz"  # 원하는 voice_id 설정

        for translation_id, text, start_time in translations:
            try:
                # ✅ Voice 객체를 사용하여 voice_id 지정
                voice = Voice(voice_id=selected_voice_id)

                # ✅ 올바른 generate() 호출 방식 적용
                audio = generate(
                    text=text,
                    voice=voice,  # voice_id를 Voice 객체로 전달
                    model="eleven_multilingual_v2"
                )

                # ✅ TTS 파일 저장
                tts_audio_path = os.path.join(tts_output_dir, f"{translation_id}.mp3")
                with open(tts_audio_path, "wb") as tts_file:
                    tts_file.write(audio)

                # ✅ 음성 파일 길이(duration) 계산
                duration = librosa.get_duration(path=tts_audio_path)

                # ✅ DB에 저장 (start_time은 transcripts에서 가져온 값 사용)
                curs.execute("""
                    INSERT INTO tts (translation_id, file_path, voice, start_time, duration)
                    VALUES (%s, %s, %s, %s, %s);
                """, (translation_id, tts_audio_path, selected_voice_id, float(start_time), float(duration)))

            except Exception as e:
                logging.error(f"TTS 생성 실패: {str(e)}")

        conn.commit()
        curs.close()
        conn.close()

        logging.info(f"TTS 생성 완료: video_id={video_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {str(e)}")
    

#########################
# 📌 6. 결과물 전달 
#########################
async def get_edit_data(video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()

        # 🎥 비디오 정보 가져오기
        curs.execute("SELECT video_id, file_name, file_path, duration FROM videos WHERE video_id = %s;", (video_id,))
        video = curs.fetchone()
        if not video:
            raise HTTPException(status_code=404, detail="해당 비디오를 찾을 수 없습니다.")

        video_data = {
            "video_id": video[0],
            "file_name": video[1],
            "file_path": video[2],
            "duration": float(video[3])  # np.float64 변환
        }

        # 🎼 배경음 정보 가져오기 (배경음이 있을 경우)
        curs.execute("SELECT file_path, volume FROM background_music WHERE video_id = %s;", (video_id,))
        bgm = curs.fetchone()
        background_music = {
            "file_path": bgm[0] if bgm else None,
            "volume": float(bgm[1]) if bgm else 1.0  # 기본 볼륨 1.0
        }

        # 🎙️ TTS 트랙 정보 가져오기 (번역된 텍스트 포함)
        curs.execute("""
            SELECT t.tts_id, t.file_path, t.voice, t.start_time, t.duration, tr.text
            FROM tts t
            JOIN translations tr ON t.translation_id = tr.translation_id
            JOIN transcripts ts ON tr.transcript_id = ts.transcript_id
            WHERE ts.video_id = %s;
        """, (video_id,))
        
        tts_tracks = [
            {
                "tts_id": row[0],
                "file_path": row[1],
                "voice": row[2],
                "start_time": float(row[3]),  # np.float64 변환
                "duration": float(row[4]),
                "translated_text": row[5]  # ✅ 번역된 텍스트 추가
            }
            for row in curs.fetchall()
        ]

        conn.close()

        # ✅ 최종 JSON 데이터 (번역된 텍스트 포함)
        response_data = {
            "video": video_data,
            "background_music": background_music,
            "tts_tracks": tts_tracks
        }

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 조회 실패: {str(e)}")
    
