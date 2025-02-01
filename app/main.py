import os
import openai
import psycopg2
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from whisper import load_model
from moviepy.editor import AudioFileClip
import logging
from elevenlabs import ElevenLabs

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

OPENAI_API_KEY = "token key"
openai.api_key = OPENAI_API_KEY

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
# 초기화 라우트
#########################
@app.get("/init")
def init_db():
    """
    users 테이블 생성 + 샘플 데이터 삽입
    """
    conn = get_connection()
    curs = conn.cursor()

    # 테이블 생성 (PostgreSQL에서 SERIAL 사용)
    curs.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        );
    """)

    # 샘플 데이터 삽입 (중복 에러 방지)
    try:
        curs.execute("INSERT INTO users (username, password) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING;", ('Alice', 'alice123'))
        curs.execute("INSERT INTO users (username, password) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING;", ('Bob', 'bob123'))
    except Exception as e:
        print(f"데이터 삽입 오류: {e}")

    conn.commit()
    curs.close()
    conn.close()
    return {"message": "DB initialized and sample data inserted"}

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

#########################
# 동영상 업로드 라우트
#########################
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    클라이언트에서 동영상을 업로드하는 엔드포인트
    """
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return JSONResponse(content={"message": f"파일 {file.filename} 업로드 완료"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#########################
# STT + 번역 기능
#########################
@app.post("/stt-video")
async def upload_and_transcribe(file: UploadFile = File(...)):
    """
    동영상을 업로드받아 Whisper로 STT 수행 후 ChatGPT로 번역
    """
    try:
        logging.info("STT 및 번역 작업 시작")

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        audio_path = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(file.filename)[0]}.wav")
        audio_clip = AudioFileClip(video_path)
        audio_clip.write_audiofile(audio_path)
        audio_clip.close()

        logging.info("Whisper 대본 생성 시작")
        result = model.transcribe(audio_path, word_timestamps=True)

        transcription_file = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(file.filename)[0]}_transcription.txt")
        with open(transcription_file, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")

        return JSONResponse(content={"transcription": open(transcription_file, "r", encoding="utf-8").read()}, status_code=200)

    except Exception as e:
        logging.error(f"STT 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")
