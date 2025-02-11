import os
import time
import openai
import psycopg2
import librosa
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from whisper import load_model
from moviepy.editor import VideoFileClip
from spleeter.separator import Separator
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

# PostgreSQL 설정
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

# FastAPI 앱 생성
app = FastAPI()

# 정적 파일 제공 (예: 영상 파일과 추출된 오디오 파일을 제공)
app.mount("/videos", StaticFiles(directory="uploaded_videos"), name="videos")
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

# 정적 파일 제공
UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
AUDIO_FOLDER = "extracted_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Pydantic 모델 (사용자 관련 예시)
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    user_id: int
    username: str

# Whisper 모델 로드 (STT 처리용)
WHISPER_MODEL = "large"
model = load_model(WHISPER_MODEL)

# OpenAI API 설정 (번역용)
OPENAI_API_KEY = "gpt-key"
openai.api_key = OPENAI_API_KEY

# PostgreSQL 연결 함수
def get_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

# 루트 라우트
@app.get("/")
def read_root():
    return {"message": "Hello from Service A (STT/Translation)!"}

# 사용자 관련 엔드포인트들 (생략 가능)
@app.post("/users", response_model=User)
def create_user(data: UserCreate):
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id FROM users WHERE username = %s;", (data.username,))
    row = curs.fetchone()
    if row:
        curs.close()
        conn.close()
        raise HTTPException(status_code=400, detail="이미 사용 중인 사용자 이름입니다.")
    curs.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING user_id;", (data.username, data.password))
    new_id = curs.fetchone()[0]
    conn.commit()
    curs.close()
    conn.close()
    return {"user_id": new_id, "username": data.username}

@app.get("/users", response_model=list[User])
def list_users():
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id, username FROM users;")
    rows = curs.fetchall()
    curs.close()
    conn.close()
    return [{"user_id": r[0], "username": r[1]} for r in rows]

@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int):
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id, username FROM users WHERE user_id = %s;", (user_id,))
    row = curs.fetchone()
    curs.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"user_id": row[0], "username": row[1]}

# Spleeter 결과 폴더에서 vocals.wav와 accompaniment.wav 찾기 함수
def find_spleeter_output(base_folder: str, file_name: str):
    expected_folder = os.path.join(base_folder, f"{file_name}_audio")
    if not os.path.exists(expected_folder):
        raise FileNotFoundError(f"❌ {expected_folder} 경로가 존재하지 않습니다!")
    for root, dirs, files in os.walk(expected_folder):
        if "vocals.wav" in files and "accompaniment.wav" in files:
            return os.path.join(root, "vocals.wav"), os.path.join(root, "accompaniment.wav")
    raise FileNotFoundError(f"❌ '{expected_folder}' 내부에 vocals.wav 또는 accompaniment.wav를 찾을 수 없습니다!")

# 오디오 분할 함수 (10MB 이하 단위)
def split_audio(input_path: str, output_dir: str, max_size_mb: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    try:
        audio = AudioSegment.from_file(input_path)
        total_length_ms = len(audio)
        bitrate = 192000  # 192kbps
        max_duration_sec = (max_size_mb * 8000000) / bitrate
        max_duration_ms = int(max_duration_sec * 1000)
        parts = []
        for i in range(0, total_length_ms, max_duration_ms):
            part = audio[i : i + max_duration_ms]
            part_path = os.path.join(output_dir, f"part_{len(parts)}.mp3")
            part.export(part_path, format="mp3", bitrate="192k")
            parts.append(part_path)
        print(f"🔹 분할된 파일 개수: {len(parts)}")
        return parts
    except Exception as e:
        print(f"❌ 오디오 분할 실패: {str(e)}")
        return []

# 영상 업로드 및 처리 엔드포인트  
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_name = os.path.splitext(file.filename)[0]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        # 기존 처리 결과 삭제
        extracted_audio_subfolder = os.path.join(AUDIO_FOLDER, f"{file_name}_audio")
        if os.path.exists(extracted_audio_subfolder):
            shutil.rmtree(extracted_audio_subfolder, ignore_errors=True)
        # 영상 파일 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # 영상 길이 계산
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        # 오디오 추출
        extracted_audio_path = os.path.join(AUDIO_FOLDER, f"{file_name}_audio.mp3")
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(extracted_audio_path, codec='mp3')
        audio_clip.close()
        video_clip.close()
        # Spleeter 실행
        try:
            separator = Separator("spleeter:2stems")
            separator.separate_to_file(extracted_audio_path, AUDIO_FOLDER)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Spleeter 실행 실패: {str(e)}")
        # vocals.wav 및 배경음 찾기
        vocals_path, bgm_path = find_spleeter_output(AUDIO_FOLDER, file_name)
        fixed_vocals_path = os.path.join(AUDIO_FOLDER, f"{file_name}_vocals.wav")
        fixed_bgm_path = os.path.join(AUDIO_FOLDER, f"{file_name}_bgm.wav")
        shutil.move(vocals_path, fixed_vocals_path)
        shutil.move(bgm_path, fixed_bgm_path)
        shutil.rmtree(os.path.join(AUDIO_FOLDER, f"{file_name}_audio"), ignore_errors=True)
        # DB에 비디오 정보 저장
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "INSERT INTO videos (file_name, file_path, duration) VALUES (%s, %s, %s) RETURNING video_id;",
            (file.filename, file_path, duration)
        )
        video_id = curs.fetchone()[0]
        curs.execute(
            "INSERT INTO background_music (video_id, file_path, volume) VALUES (%s, %s, %s);",
            (video_id, fixed_bgm_path, 1.0)
        )
        conn.commit()
        curs.close()
        conn.close()
        # STT 및 번역 처리 (서비스 A에서 처리)
        await transcribe_audio(fixed_vocals_path, video_id)
        await translate_video(video_id)
        
        # << 변경된 부분 >>
        # 서비스 A는 TTS 생성을 직접 수행하지 않고, STT/번역 결과를 바탕으로
        # 서비스 B의 TTS 생성 API를 호출하여 TTS를 생성하도록 요청합니다.
        stt_data = {"video_id": video_id}  # 필요한 추가 정보를 포함할 수 있음
        tts_response = requests.post("http://localhost:8001/generate-tts-from-stt", json=stt_data)
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS 생성 서비스 호출 실패")
        # << 끝 >>
        
        # 최종 결과 데이터를 조회하여 클라이언트에 전달 (영상, 오디오, TTS 결과 등)
        return await get_edit_data(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

# STT 변환 및 DB 저장
async def transcribe_audio(audio_path: str, video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"STT 변환 실패: {audio_path} 파일이 존재하지 않습니다.")
        result = model.transcribe(audio_path, word_timestamps=True)
        for segment in result["segments"]:
            start_time = float(segment["start"])
            end_time = float(segment["end"])
            curs.execute(
                """
                INSERT INTO transcripts (video_id, language, text, start_time, end_time)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (video_id, "ko", segment["text"], start_time, end_time)
            )
        conn.commit()
        curs.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")

# 번역 및 DB 저장
async def translate_video(video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("SELECT transcript_id, text FROM transcripts WHERE video_id = %s;", (video_id,))
        transcripts = curs.fetchall()
        if not transcripts:
            raise HTTPException(status_code=404, detail="STT 데이터가 없습니다.")
        for transcript_id, text in transcripts:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Translate the following Korean text into English."},
                    {"role": "user", "content": text}
                ]
            )
            translated_text = response["choices"][0]["message"]["content"].strip()
            curs.execute(
                """
                INSERT INTO translations (transcript_id, language, text)
                VALUES (%s, %s, %s);
                """,
                (transcript_id, "en", translated_text)
            )
        conn.commit()
        curs.close()
        conn.close()
        print(f"번역 완료: video_id={video_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"번역 실패: {str(e)}")

# 최종 결과 조회 엔드포인트
async def get_edit_data(video_id: int):
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("SELECT video_id, file_name, file_path, duration FROM videos WHERE video_id = %s;", (video_id,))
        video = curs.fetchone()
        if not video:
            raise HTTPException(status_code=404, detail="해당 비디오를 찾을 수 없습니다.")
        video_data = {
            "video_id": video[0],
            "file_name": video[1],
            "file_path": video[2],
            "duration": float(video[3])
        }
        curs.execute("SELECT file_path, volume FROM background_music WHERE video_id = %s;", (video_id,))
        bgm = curs.fetchone()
        background_music = {
            "file_path": bgm[0] if bgm else None,
            "volume": float(bgm[1]) if bgm else 1.0
        }
        curs.execute(
            """
            SELECT t.tts_id, t.file_path, t.voice, t.start_time, t.duration, tr.text
            FROM tts t
            JOIN translations tr ON t.translation_id = tr.translation_id
            JOIN transcripts ts ON tr.transcript_id = ts.transcript_id
            WHERE ts.video_id = %s;
            """,
            (video_id,)
        )
        tts_tracks = [
            {
                "tts_id": row[0],
                "file_path": row[1],
                "voice": row[2],
                "start_time": float(row[3]),
                "duration": float(row[4]),
                "translated_text": row[5]
            }
            for row in curs.fetchall()
        ]
        conn.close()
        response_data = {
            "video": video_data,
            "background_music": background_music,
            "tts_tracks": tts_tracks
        }
        return JSONResponse(content=response_data, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 조회 실패: {str(e)}")
