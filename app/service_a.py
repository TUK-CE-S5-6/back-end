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
from pyannote.audio import Pipeline

# PostgreSQL 설정
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

# FastAPI 앱 생성
app = FastAPI()

# 정적 파일 제공 (영상 파일과 추출된 오디오 파일)
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

# 폴더 생성
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

# 사용자 관련 엔드포인트들
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

# -----------------------------------------------------------------
# EEND 기반 화자 다이어리제이션 함수 (pyannote.audio 사용)
# -----------------------------------------------------------------
def diarize_audio_eend(input_path: str):
    """
    입력 오디오 파일(input_path)을 받아 화자 다이어리제이션을 수행한 후,
    각 세그먼트의 시작/종료 시간과 화자 라벨을 포함한 리스트를 반환합니다.
    반환 형식 예시:
      [
          {"start": 0.0, "end": 3.0, "speaker": "speaker_0"},
          {"start": 3.0, "end": 6.0, "speaker": "speaker_1"},
          ...
      ]
    """
    # pyannote.audio는 wav 파일을 권장하므로 mp3인 경우 wav로 변환
    if not input_path.lower().endswith('.wav'):
        wav_path = os.path.splitext(input_path)[0] + '.wav'
        sound = AudioSegment.from_file(input_path)
        sound.export(wav_path, format="wav")
    else:
        wav_path = input_path

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(wav_path)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments

# -----------------------------------------------------------------
# STT 변환 (화자 다이어리제이션 + Whisper STT 결합)
# -----------------------------------------------------------------
async def transcribe_audio(audio_path: str, video_id: int):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"STT 변환 실패: {audio_path} 파일이 존재하지 않습니다.")
        
        # 화자 다이어리제이션 수행 (EEND 기반, pyannote.audio 사용)
        segments = diarize_audio_eend(audio_path)
        
        # 전체 오디오 로드 (pydub 사용)
        audio = AudioSegment.from_file(audio_path)
        os.makedirs("temp_segments", exist_ok=True)
        
        conn = get_connection()
        curs = conn.cursor()
        for seg in segments:
            start_sec = seg["start"]
            end_sec = seg["end"]
            speaker = seg["speaker"]
            duration_sec = end_sec - start_sec
            
            # 짧은 세그먼트(예: 0.5초 미만)는 스킵
            if duration_sec < 0.5:
                continue

            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            # 해당 구간의 오디오 추출 및 임시 저장
            segment_audio = audio[start_ms:end_ms]
            temp_segment_path = os.path.join("temp_segments", f"{video_id}_{start_ms}_{end_ms}.mp3")
            segment_audio.export(temp_segment_path, format="mp3")
            
            # Whisper를 이용한 STT 수행
            result = model.transcribe(temp_segment_path, word_timestamps=True)
            # 반환 결과의 타입에 따라 text 추출
            if isinstance(result, dict):
                text = result.get("text", "").strip()
            elif isinstance(result, tuple) and len(result) > 0:
                text = result[0].strip()
            else:
                text = ""
            
            # DB에 화자 정보와 함께 전사 결과 저장
            curs.execute(
                """
                INSERT INTO transcripts (video_id, language, text, start_time, end_time, speaker)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (video_id, "ko", text, start_sec, end_sec, speaker)
            )
            
            os.remove(temp_segment_path)
        conn.commit()
        curs.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {str(e)}")


# -----------------------------------------------------------------
# 번역 및 DB 저장
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# 최종 결과 조회 엔드포인트
# -----------------------------------------------------------------
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
        # 기존 쿼리에 transcripts의 speaker 컬럼 추가 (ts.speaker)
        curs.execute(
            """
            SELECT t.tts_id, t.file_path, t.voice, t.start_time, t.duration, tr.text, ts.speaker
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
                "translated_text": row[5],
                "speaker": row[6]
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

# -----------------------------------------------------------------
# 영상 업로드 및 처리 엔드포인트  
# -----------------------------------------------------------------
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # 업로드 파일명에서 확장자 제거
        original_file_name = file.filename  # 예: videoplayback33.mp4
        file_name = os.path.splitext(original_file_name)[0]
        # 이미 "_audio"가 포함되어 있다면 제거하여 기본 이름으로 사용
        base_name = file_name[:-len("_audio")] if file_name.endswith("_audio") else file_name

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        # 기존 처리 결과 삭제 (Spleeter 출력 폴더)
        extracted_audio_subfolder = os.path.join(AUDIO_FOLDER, f"{base_name}_audio")
        if os.path.exists(extracted_audio_subfolder):
            shutil.rmtree(extracted_audio_subfolder, ignore_errors=True)
        # 영상 파일 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # 영상 길이 계산 및 오디오 추출
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        # 오디오 추출 파일명은 base_name + ".mp3"로 저장 → Spleeter는 이를 기준으로 하위 폴더 생성
        extracted_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(extracted_audio_path, codec='mp3')
        audio_clip.close()
        video_clip.close()

        # Spleeter 처리는 Service B의 /separate-audio 엔드포인트를 호출하여 처리
        with open(extracted_audio_path, "rb") as audio_file:
            separation_response = requests.post(
                "http://localhost:8001/separate-audio",
                files={"file": audio_file}
            )
        if separation_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Spleeter 분리 서비스 호출 실패")
        separation_data = separation_response.json()
        
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
            (video_id, separation_data.get("bgm_path"), 1.0)
        )
        conn.commit()
        curs.close()
        conn.close()

        # STT 및 번역 처리 (화자 다이어리제이션 + STT)
        await transcribe_audio(separation_data.get("vocals_path"), video_id)
        await translate_video(video_id)
        
        # TTS 생성은 Service B의 엔드포인트를 호출하여 처리
        stt_data = {"video_id": video_id}
        tts_response = requests.post("http://localhost:8001/generate-tts-from-stt", json=stt_data)
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS 생성 서비스 호출 실패")
        
        # 최종 결과 조회 후 클라이언트에 전달
        return await get_edit_data(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")