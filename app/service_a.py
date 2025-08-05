import os
import time
import json
import openai
import psycopg2
import requests
import asyncio
import shutil
import uuid
import tempfile
from typing import List
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Response, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from pyannote.audio import Pipeline
from moviepy.editor import (
    VideoFileClip, 
    AudioFileClip, 
    CompositeVideoClip, 
    CompositeAudioClip,
    concatenate_videoclips
)
from typing import Dict
import moviepy.video.fx.all as vfx

BASE_HOST = "http://175.116.3.178"  # 또는 배포 시 EC2 인스턴스의 공인 IP나 도메인

SPLITTER_HOST = "http://localhost:8001"
API_HOST      = "http://localhost:8001"

# PostgreSQL 설정 (환경에 맞게 수정)
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

# Clova Speech Long Sentence API 설정
NAVER_CLOVA_SECRET_KEY = "clova-key"  
NAVER_CLOVA_SPEECH_URL = "invoke-url"

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

# FastAPI 앱 생성
app = FastAPI()

#앱 루트 디렉토리
BASE_DIR = os.getcwd()

STATIC_MAP = {
    '/uploaded_videos/': os.path.join(os.getcwd(), 'uploaded_videos'),
    '/extracted_audio/': os.path.join(os.getcwd(), 'extracted_audio'),
    '/user_files/': os.path.join(os.getcwd(), 'user_files'),
}
LOCAL_HOSTS = {"175.116.3.178:8000", "localhost:8000", "127.0.0.1:8000"}

#진행률 계산
PROGRESS = {}       # job_id → percent
cumulative_pct = {
   'upload_time': 0,
   'audio_extraction_time': 10,
   'spleeter_time': 25,
   'db_time': 38,
   'stt_time': 46,
   'translation_time': 64,
   'tts_time': 88,
   'get_time': 100
}

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 제공 (영상 파일과 추출된 오디오 파일)
app.mount("/uploaded_videos", StaticFiles(directory="uploaded_videos"), name="videos")
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")
app.mount("/user_files", StaticFiles(directory="user_files"), name="user_files")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")


# 폴더 생성
UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
AUDIO_FOLDER = "extracted_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# 사용자별 파일 관리를 위한 폴더 경로 설정
USER_FILES_FOLDER = "user_files"
os.makedirs(USER_FILES_FOLDER, exist_ok=True)

THUMBNAIL_FOLDER = "thumbnails"
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

# Pydantic 모델 (사용자 관련 예시)
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    user_id: int
    username: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_progress(self, job_id: str, progress: int):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json({"progress": progress})
            except WebSocketDisconnect:
                self.disconnect(job_id)

ws_manager = ConnectionManager()

class MediaItem(BaseModel):
    path: str
    start_time: float

class MergePayload(BaseModel):
    videos: list[MediaItem]
    audios: list[MediaItem]
    
class SubtitleSegment(BaseModel):
    id: int
    start_time: float
    end_time: float
    text: str

############################################
# 회원가입 엔드포인트 (토큰을 JSON에 포함)
############################################
@app.post("/signup")
async def signup(response: Response, username: str = Form(...), password: str = Form(...)):
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        if curs.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        curs.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING user_id", (username, password))
        user_id = curs.fetchone()[0]
        conn.commit()
        curs.close()
        conn.close()
        token = f"token-for-user-{user_id}"
        return JSONResponse(content={"message": "Signup successful", "user_id": user_id, "token": token})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

############################################
# 로그인 엔드포인트 (토큰을 JSON에 포함)
############################################
@app.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("SELECT user_id, password FROM users WHERE username = %s", (username,))
        result = curs.fetchone()
        if result is None or result[1] != password:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        user_id = result[0]
        curs.close()
        conn.close()
        token = f"token-for-user-{user_id}"
        return JSONResponse(content={"message": "Login successful", "user_id": user_id, "token": token})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

############################################
# 사용자 인증 유틸 함수 (Authorization 헤더 사용)
############################################
def get_current_user_id(request: Request) -> int:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.split(" ")[1]
    try:
        return int(token.split("-")[-1])
    except:
        raise HTTPException(status_code=401, detail="Invalid token format")
    
def get_user_id_from_token(request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.replace("Bearer ", "")
    try:
        user_id = int(token.split("-")[-1])
        return user_id
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")


############################################
# 현재 로그인한 사용자 정보
############################################
@app.get("/me")
async def get_current_user(request: Request):
    user_id = get_current_user_id(request)
    return {"user_id": user_id}

############################################
# 프로젝트 목록
############################################
@app.get("/projects")
async def get_projects(request: Request):
    user_id = get_current_user_id(request)
    conn = get_connection()
    curs = conn.cursor()
    # 프로젝트와 연결된 최소 video_id(=첫 영상)를 가져오도록 JOIN
    curs.execute("""
        SELECT p.project_id,
               p.project_name,
               p.description,
               p.created_at,
               MIN(v.video_id) AS first_video_id
        FROM projects p
        LEFT JOIN videos v
          ON v.project_id = p.project_id
        WHERE p.user_id = %s
        GROUP BY p.project_id, p.project_name, p.description, p.created_at
        ORDER BY p.created_at DESC;
    """, (user_id,))
    rows = curs.fetchall()
    curs.close()
    conn.close()

    projects = []
    for project_id, name, desc, created_at, first_video_id in rows:
        projects.append({
            "project_id": project_id,
            "project_name": name,
            "description": desc,
            "created_at": created_at.isoformat() if created_at else None,
            "video_id": first_video_id  # 여기 추가
        })

    return JSONResponse(content={"projects": projects})


############################################
# 프로젝트 추가
############################################
@app.post("/projects/add")
async def add_project(
    request: Request,
    project_name: str = Form(...),
    description: str = Form(""),
    source_language: str = Form("ko-KR"),
    target_language: str = Form("en-US")
):
    user_id = get_current_user_id(request)
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("""
            INSERT INTO projects (project_name, description, user_id, source_language, target_language)
            VALUES (%s, %s, %s, %s, %s) RETURNING project_id
        """, (project_name, description, user_id, source_language, target_language))
        project_id = curs.fetchone()[0]
        conn.commit()
        curs.close()
        conn.close()
        return JSONResponse(content={
            "project_id": project_id,
            "project_name": project_name,
            "description": description,
            "source_language": source_language,
            "target_language": target_language
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

############################################
# 프로젝트 삭제
############################################
@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, request: Request):
    user_id = get_current_user_id(request)
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("SELECT project_id FROM projects WHERE project_id = %s AND user_id = %s", (project_id, user_id))
        if not curs.fetchone():
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없거나 권한이 없습니다.")
        curs.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))
        conn.commit()
        curs.close()
        conn.close()
        return JSONResponse(content={"message": "프로젝트 삭제 성공"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/videos/edit_data")
async def get_project_videos_edit_data(project_id: int, request: Request):
    user_id = get_user_id_from_token(request)
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "SELECT project_id FROM projects WHERE project_id = %s AND user_id = %s",
            (project_id, user_id)
        )
        if curs.fetchone() is None:
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없거나 권한이 없습니다.")
        
        curs.execute("SELECT video_id FROM videos WHERE project_id = %s", (project_id,))
        video_ids = [row[0] for row in curs.fetchall()]
        curs.close()
        conn.close()
        
        videos_data = []
        for vid in video_ids:
            response = await get_edit_data(vid)
            if isinstance(response.body, bytes):
                data = json.loads(response.body.decode())
            else:
                data = response.body
            videos_data.append(data)
        
        return JSONResponse(content={"videos": videos_data}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def detect_file_type(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    if ext in [".mp3", ".wav", ".aac"]:
        return "audio"
    elif ext in [".mp4", ".mov", ".avi", ".webm"]:
        return "video"
    elif ext in [".jpg", ".jpeg", ".png", ".gif"]:
        return "image"
    else:
        return "other"

def get_duration(file_path, file_type):
    try:
        if file_type == "audio":
            return AudioFileClip(file_path).duration
        elif file_type == "video":
            return VideoFileClip(file_path).duration
    except:
        return None
    return None

# 1. 사용자별 파일 업로드 & 썸네일 생성
@app.post("/upload-file")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # 사용자 인증에서 user_id 추출
    user_id = get_user_id_from_token(request)

    # 유저 폴더 생성
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    # 파일 저장
    save_path = os.path.join(user_folder, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 썸네일 폴더
    os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)
    base_name, ext = os.path.splitext(file.filename)
    thumb_name = f"{base_name}_thumbnail.jpg"
    thumb_path = os.path.join(THUMBNAIL_FOLDER, thumb_name)

    file_type = detect_file_type(file.filename)
    try:
        if file_type == "video":
            # 첫 프레임으로 썸네일 생성
            clip = VideoFileClip(save_path)
            clip.save_frame(thumb_path, t=0)
            clip.close()
        elif file_type == "image":
            # 이미지 파일일 땐 원본 복사
            from shutil import copyfile
            copyfile(save_path, thumb_path)
        else:
            # 기타 파일은 placeholder 사용
            placeholder = "static/file-placeholder.jpg"
            from shutil import copyfile
            copyfile(placeholder, thumb_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"썸네일 생성 실패: {e}")

    return JSONResponse(
        content={
            "file_url": f"/user_files/{user_id}/{file.filename}",
            "thumbnail_url": f"/thumbnails/{thumb_name}"
        },
        status_code=201
    )

# 2. 사용자별 파일 목록 조회
@app.get("/user-files")
async def list_user_files(request: Request):
    user_id = get_user_id_from_token(request)
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    if not os.path.exists(user_folder):
        return {"files": []}

    files_info = []
    for name in os.listdir(user_folder):
        file_path = os.path.join(user_folder, name)
        if os.path.isfile(file_path):
            base_name, _ = os.path.splitext(name)
            files_info.append({
                "file_name": name,
                "file_url": f"/user_files/{user_id}/{name}",
                "thumbnail_url": f"/thumbnails/{base_name}_thumbnail.jpg"
            })
    return {"files": files_info}

# 3. 사용자별 파일 삭제
@app.delete("/user-files")
async def delete_user_file(request: Request, file: str = Query(...)):
    user_id = get_user_id_from_token(request)
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    file_path = os.path.join(user_folder, file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    os.remove(file_path)

    # 썸네일도 제거
    base_name, _ = os.path.splitext(file)
    thumb_path = os.path.join(THUMBNAIL_FOLDER, f"{base_name}_thumbnail.jpg")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)

    return {"detail": "파일 삭제 성공"}

# 인증 헬퍼 함수 (예시)
def get_user_id_from_token(request: Request) -> int:
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return int(auth.split("-")[-1])


# 4. 사용자별 파일 다운로드 엔드포인트
@app.get("/download-file")
async def download_file(request: Request, file_name: str = Query(...)):
    user_id = get_user_id_from_token(request)
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    file_path = os.path.join(user_folder, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일이 존재하지 않습니다.")
    return FileResponse(file_path, filename=file_name, media_type="application/octet-stream")

############################################
# Clova Speech Long Sentence STT 호출 함수 (Secret Key 사용)
############################################
def clova_speech_stt(file_path: str, completion="sync", language="ko-KR", 
                     wordAlignment=True, fullText=True,
                     speakerCountMin=-1, speakerCountMax=-1) -> dict:
    """
    주어진 오디오 파일(file_path)을 Clova Speech Long Sentence API를 통해 전송하여,
    인식 결과 전체 JSON을 반환합니다.
    """
    diarization = {
        "enable": True,
        "speakerCountMin": 1,
        "speakerCountMax": speakerCountMax
    }
    request_body = {
        "language": language,
        "completion": completion,
        "wordAlignment": wordAlignment,
        "fullText": fullText,
        "diarization": diarization
    }
    headers = {
        "Accept": "application/json; charset=UTF-8",
        "X-CLOVASPEECH-API-KEY": NAVER_CLOVA_SECRET_KEY
    }
    files = {
        "media": open(file_path, "rb"),
        "params": (None, json.dumps(request_body).encode("UTF-8"), "application/json")
    }
    url = NAVER_CLOVA_SPEECH_URL + "/recognizer/upload"
    response = requests.post(url, headers=headers, files=files)
    files["media"].close()
    if response.status_code != 200:
        print("Clova Speech Long Sentence API 호출 실패:", response.status_code, response.text)
        return {}
    result = response.json()
    return result

############################################
# STT 변환 (Clova Speech Long Sentence STT 사용, 화자 인식 포함)
# 이제 클라이언트가 전달한 source_language 값을 사용합니다.
############################################
async def transcribe_audio(audio_path: str, video_id: int, source_language: str):
    step_start = time.time()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"STT 변환 실패: {audio_path} 파일이 존재하지 않습니다.")
    
    result = clova_speech_stt(
        audio_path,
        completion="sync",
        language=source_language,  # 클라이언트가 보낸 언어 코드 사용 (예: "ko-KR", "en-US", "ja", 등)
        wordAlignment=True,
        fullText=True,
        speakerCountMin=1,
        speakerCountMax=3
    )
    if not result:
        raise HTTPException(status_code=500, detail="Clova Speech STT 호출 실패")
    
    conn = get_connection()
    curs = conn.cursor()
    if "segments" in result:
        for seg in result["segments"]:
            start_sec = seg.get("start", 0) / 1000.0
            end_sec = seg.get("end", 0) / 1000.0
            text = seg.get("text", "")
            speaker_info = seg.get("speaker", {})
            speaker = speaker_info.get("name", speaker_info.get("label", ""))
            curs.execute(
                """
                INSERT INTO transcripts (video_id, language, text, start_time, end_time, speaker)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (video_id, source_language, text, start_sec, end_sec, speaker)
            )
    else:
        curs.execute(
            """
            INSERT INTO transcripts (video_id, language, text, start_time, end_time, speaker)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (video_id, source_language, result.get("text", ""), 0, 0, "")
        )
    conn.commit()
    curs.close()
    conn.close()

    stt_time = time.time() - step_start
    return {"stt_time": stt_time}

############################################
# 번역 및 DB 저장
# 클라이언트가 전달한 source_language와 target_language(모두 언어 코드)를 사용합니다.
############################################
async def translate_video(video_id: int, source_language: str, target_language: str):
    step_start = time.time()
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT transcript_id, text FROM transcripts WHERE video_id = %s;", (video_id,))
    transcripts = curs.fetchall()
    if not transcripts:
        raise HTTPException(status_code=404, detail="STT 데이터가 없습니다.")
    for transcript_id, text in transcripts:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Please translate the text from {source_language} to {target_language}, ensuring that the syllable count and sentence structure are as close to the original as possible, for TTS synchronization."},
                {"role": "user", "content": text}
            ]
        )
        translated_text = response["choices"][0]["message"]["content"].strip()
        curs.execute(
            """
            INSERT INTO translations (transcript_id, language, text)
            VALUES (%s, %s, %s);
            """,
            (transcript_id, target_language, translated_text)
        )
    conn.commit()
    curs.close()
    conn.close()
    return {"translation_time": time.time() - step_start}

############################################
# 최종 결과 조회 엔드포인트
############################################
async def get_edit_data(video_id: int):
    step_start = time.time()
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
        SELECT t.tts_id, t.file_path, t.voice, t.start_time, t.duration, 
               tr.text as translated_text, ts.text as original_text, ts.speaker
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
            "original_text": row[6],
            "speaker": row[7]
        }
        for row in curs.fetchall()
    ]
    conn.close()
    total_get_time = time.time() - step_start
    return JSONResponse(content={
        "video": video_data,
        "background_music": background_music,
        "tts_tracks": tts_tracks,
        "get_time": total_get_time
    }, status_code=200)

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id)

# 동기 블로킹 작업을 워커 스레드로 옮기는 유틸 함수
async def to_thread(func, /, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))

# 비동기 핸들러
@app.post("/upload-video")
async def upload_video(
    job_id: str = Query(...),
    file: UploadFile = File(...),
    source_language: str = Form("ko-KR"),
    target_language: str = Form("en-US"),
    project_id: int = Form(...),
):
    overall_start = time.time()
    try:
        # 1. 파일 저장
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        content = await file.read()
        await to_thread(lambda: open(file_path, 'wb').write(content))
        await ws_manager.send_progress(job_id, cumulative_pct['upload_time'])
        await asyncio.sleep(1)

        # 2. 오디오 추출
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        extracted_audio_path = os.path.join(
            AUDIO_FOLDER, f"{os.path.splitext(file.filename)[0]}.mp3"
        )
        await to_thread(
            video_clip.audio.write_audiofile,
            extracted_audio_path,
            codec="mp3"
        )
        video_clip.close()
        await ws_manager.send_progress(job_id, cumulative_pct['audio_extraction_time'])
        await asyncio.sleep(1)

        # 3. Spleeter 분리
        def call_spleeter(path):
            with open(path, 'rb') as f:
                r = requests.post(f"{SPLITTER_HOST}/separate-audio", files={'file': f})
            r.raise_for_status()
            return r.json()
        separation_data = await to_thread(call_spleeter, extracted_audio_path)
        await ws_manager.send_progress(job_id, cumulative_pct['spleeter_time'])
        await asyncio.sleep(1)

        # 4. DB 저장
        def save_video(file_name, path, dur, proj_id):
            conn = get_connection()
            curs = conn.cursor()
            curs.execute(
                "INSERT INTO videos (file_name, file_path, duration, project_id) VALUES (%s, %s, %s, %s) RETURNING video_id;",
                (file_name, path, dur, proj_id)
            )
            vid = curs.fetchone()[0]
            curs.execute(
                "INSERT INTO background_music (video_id, file_path, volume) VALUES (%s, %s, %s);",
                (vid, separation_data.get('bgm_path'), 1.0)
            )
            conn.commit(); curs.close(); conn.close()
            return vid
        video_id = await to_thread(save_video, file.filename, file_path, duration, project_id)
        await ws_manager.send_progress(job_id, cumulative_pct['db_time'])
        await asyncio.sleep(1)

        # 썸네일 생성 (첫 프레임)
        clip = VideoFileClip(file_path)
        thumb_dir = "thumbnails"
        os.makedirs(thumb_dir, exist_ok=True)
        thumb_path = os.path.join(thumb_dir, f"{video_id}.jpg")

        # t=0초(또는 원하는 시간)를 지정해서 프레임을 저장
        clip.save_frame(thumb_path, t=0)

        clip.close()

        # 5. STT (클로바)
        stt_timings = await transcribe_audio(
            separation_data.get('vocals_path'),
            video_id,
            source_language
        )
        await ws_manager.send_progress(job_id, cumulative_pct['stt_time'])
        await asyncio.sleep(1)

        # 6. 번역
        translation_timings = await translate_video(
            video_id, source_language, target_language
        )
        await ws_manager.send_progress(job_id, cumulative_pct['translation_time'])
        await asyncio.sleep(1)

        # 7. TTS 생성
        step_start = time.time()
        await to_thread(
            lambda vid: requests.post(f"{API_HOST}/generate-tts-from-stt", json={'video_id': vid}).raise_for_status(),
            video_id
        )
        await ws_manager.send_progress(job_id, cumulative_pct['tts_time'])
        await asyncio.sleep(1)

        # 8. 최종 데이터 조회
        # 내부적으로 DB에서 최종 edit data를 구성
        _ = await get_edit_data(video_id)
        await ws_manager.send_progress(job_id, cumulative_pct['get_time'])

        return JSONResponse(content={'job_id': job_id}, status_code=202)

    except Exception as e:
        await ws_manager.send_progress(job_id, -1)
        raise HTTPException(status_code=500, detail=f"업로드 실패: {e}")
    
def resolve_local_path(url: str) -> str:
    parsed = urlparse(url)
    host = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname
    if host in LOCAL_HOSTS:
        for prefix, dirpath in STATIC_MAP.items():
            if parsed.path.startswith(prefix):
                rel = parsed.path[len(prefix):].lstrip("/")
                return os.path.join(dirpath, rel)
    return url

@app.post("/merge-media")
async def merge_media(request: Request):
    """
    JSON payload 예시:
    {
      "videoTracks": [
        {
          "name": "Video Track 1",
          "volume": 80,
          "tracks": [
            { "url": "...mp4", "startTime": 0.0 }
          ]
        },
        {
          "name": "Video Track 2",
          "volume": 100,
          "tracks": [
            { "url": "...mp4", "startTime": 2.0 }
          ]
        }
      ],
      "audioTracks": [
        {
          "volume": 50,
          "tracks": [
            { "url": "...wav", "startTime": 0.0 }
          ]
        },
        {
          "volume": 100,
          "tracks": [
            { "url": "...mp3", "startTime": 3.5 }
          ]
        }
      ]
    }
    """
    payload = await request.json()
    tempdir = tempfile.mkdtemp(prefix="merge_")
    video_items = []  # (priority, clip)
    audio_clips = []

    def fetch_and_resolve(raw_url: str) -> str:
        path = resolve_local_path(raw_url)
        if path.startswith("http"):
            r = requests.get(raw_url, stream=True)
            r.raise_for_status()
            ext = os.path.splitext(urlparse(raw_url).path)[1]
            fn = os.path.join(tempdir, f"dl_{time.time():.0f}{ext}")
            with open(fn, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            return fn
        if not os.path.isfile(path):
            raise HTTPException(404, f"파일을 찾을 수 없습니다: {path}")
        return path

    # ▶ 비디오 트랙: name에서 숫자 파싱하여 priority로 사용
    for group in payload.get("videoTracks", []):
        name = group.get("name", "")
        try:
            # "Video Track 1" → 1
            priority = int(name.strip().split()[-1])
        except:
            priority = 0
        vol = float(group.get("volume", 100)) / 100.0

        for track in group.get("tracks", []):
            url   = track.get("url")      or HTTPException(400, "video url 누락")
            start = float(track.get("startTime", 0))
            fp    = fetch_and_resolve(url)
            clip  = VideoFileClip(fp).set_start(start).volumex(vol)
            video_items.append((priority, clip))

    # ▶ 오디오 트랙: 이름 무시, volume/startTime만
    for group in payload.get("audioTracks", []):
        vol = float(group.get("volume", 100)) / 100.0
        for track in group.get("tracks", []):
            url   = track.get("url")      or HTTPException(400, "audio url 누락")
            start = float(track.get("startTime", 0))
            fp    = fetch_and_resolve(url)
            ac    = AudioFileClip(fp).set_start(start).volumex(vol)
            audio_clips.append(ac)

    if not video_items and not audio_clips:
        raise HTTPException(400, "합성할 트랙이 없습니다.")

    # ▶ priority 순 정렬 (작은 숫자 → 배경, 큰 숫자 → 전경)
    video_items.sort(key=lambda x: -x[0])
    ordered_clips = [clip for _, clip in video_items]

    # ▶ 비디오 합성
    final_video = CompositeVideoClip(
        ordered_clips,
        size=ordered_clips[0].size if ordered_clips else None
    )

    # ▶ 오디오 합성
    all_audio = [v.audio for v in ordered_clips if v.audio] + audio_clips
    if all_audio:
        final_video.audio = CompositeAudioClip(all_audio)

    # ▶ 출력
    out_path = os.path.join(tempdir, "merged_output.mp4")
    final_video.write_videofile(out_path, codec="libx264", audio_codec="aac")

    # ▶ 리소스 해제
    for _, c in video_items: c.close()
    for a in audio_clips:        a.close()

    return FileResponse(
        path=out_path,
        filename="merged_output.mp4",
        media_type="video/mp4"
    )
    
@app.post("/translate-text")
async def translate_text_endpoint(
    text: str = Form(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    ALLOWED_TARGET_LANGUAGES = {"ko-KR", "en-US", "enko", "ja", "zh-cn", "zh-tw"}
    # 대상 언어 확인
    if target_language not in ALLOWED_TARGET_LANGUAGES:
        raise HTTPException(status_code=400, detail="지원되지 않는 대상 언어입니다.")

    try:
        # 번역 프롬프트: 원본 텍스트와 최대한 길이가 유사하게 번역
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        f"Translate the following text from {source_language} to {target_language} in a concise manner "
                        "while keeping the translation as close in length as possible to the original. "
                        "Maintain nearly identical word count and sentence structure so that the final TTS output "
                        "closely matches the original timing."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        translated_text = response["choices"][0]["message"]["content"].strip()
        return JSONResponse(content={"translated_text": translated_text}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ----------------------------
# 파일 관리 엔드포인트 추가
# ----------------------------

@app.get("/list-file")
async def list_file():
    try:
        # uploaded_videos 폴더 내 파일 목록 (파일만)
        uploaded_videos_dir = "uploaded_videos"
        uploaded_files = [
            f for f in os.listdir(uploaded_videos_dir)
            if os.path.isfile(os.path.join(uploaded_videos_dir, f))
        ]

        # extracted_audio 폴더 내 특정 하위 폴더 목록
        extracted_audio_dir = "extracted_audio"
        subfolders = ["sound_effects", "custom_tts"]
        extracted_audio_structure = {}
        for folder in subfolders:
            folder_path = os.path.join(extracted_audio_dir, folder)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                # 파일만 반환
                file_list = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
                extracted_audio_structure[folder] = file_list
            else:
                extracted_audio_structure[folder] = []
        
        return JSONResponse(
            content={
                "uploaded_videos": uploaded_files,
                "extracted_audio": extracted_audio_structure
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file-details")
async def file_details(filename: str = Query(..., description="업로드된 비디오 파일명 (예: example.mp4)")):
    try:
        # DB에서 해당 영상 정보 조회 (file_name이 정확히 일치하는 레코드)
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "SELECT video_id, file_name, file_path, duration FROM videos WHERE file_name = %s ORDER BY video_id DESC LIMIT 1;",
            (filename,)
        )
        video = curs.fetchone()
        if not video:
            raise HTTPException(status_code=404, detail="해당 영상 정보를 찾을 수 없습니다.")
        
        video_id = video[0]
        base_name = os.path.splitext(video[1])[0]  # 예: "example" (확장자 제거)

        # URL 구성 (정적 파일 제공 경로에 맞게)
        video_url = f"{BASE_HOST}:8000/uploaded_videos/{video[1]}"
        extracted_audio_filename = f"{base_name}.mp3"
        extracted_audio_url = f"{BASE_HOST}:8000/extracted_audio/{extracted_audio_filename}"

        # Spleeter 결과 폴더 내의 vocal.wav와 accompaniment.wav
        vocal_path = os.path.join("extracted_audio", base_name, "vocals.wav")
        accompaniment_path = os.path.join("extracted_audio", base_name, "accompaniment.wav")
        vocal_url = f"{BASE_HOST}:8000/extracted_audio/{base_name}/vocals.wav" if os.path.exists(vocal_path) else None
        accompaniment_url = f"{BASE_HOST}:8000/extracted_audio/{base_name}/accompaniment.wav" if os.path.exists(accompaniment_path) else None

        # DB에서 TTS 트랙 정보 조회 (영상과 연결된 tts 레코드)
        curs.execute(
            """
            SELECT t.tts_id, t.file_path, t.voice, t.start_time, t.duration, 
                   tr.text AS translated_text, ts.text AS original_text, ts.speaker
            FROM tts t
            JOIN translations tr ON t.translation_id = tr.translation_id
            JOIN transcripts ts ON tr.transcript_id = ts.transcript_id
            WHERE ts.video_id = %s;
            """,
            (video_id,)
        )
        tts_rows = curs.fetchall()
        tts_tracks = []
        for row in tts_rows:
            # DB에 저장된 전체 상대경로를 그대로 사용하여 URL 구성
            tts_url = f"{BASE_HOST}:8000/{row[1]}"
            tts_tracks.append({
                "tts_id": row[0],
                "file_url": tts_url,
                "voice": row[2],
                "start_time": row[3],
                "duration": row[4],
                "translated_text": row[5],
                "original_text": row[6],
                "speaker": row[7]
            })
        curs.close()
        conn.close()

        return JSONResponse(content={
            "video": {
                "file_name": video[1],
                "url": video_url,
                "duration": float(video[3])
            },
            "extracted_audio": {
                "file_name": extracted_audio_filename,
                "url": extracted_audio_url
            },
            "spleeter": {
                "vocal": vocal_url,
                "accompaniment": accompaniment_url
            },
            "tts_tracks": tts_tracks
        }, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-video")
async def delete_video(filename: str = Query(..., description="삭제할 영상 파일명")):
    # DB에서 해당 영상 조회
    conn = get_connection()
    curs = conn.cursor()
    curs.execute(
            "SELECT video_id, file_name, file_path, duration FROM videos WHERE file_name = %s ORDER BY video_id DESC LIMIT 1;",
            (filename,)
        )
    video = curs.fetchone()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 존재하지 않습니다.")
    video_id, video_path = video
    base_name = os.path.splitext(filename)[0]

    # 파일 시스템에서 영상 파일 삭제
    if os.path.exists(video_path):
        os.remove(video_path)

    # 추출된 오디오 파일 삭제
    extracted_audio_path = os.path.join("extracted_audio", f"{base_name}.mp3")
    if os.path.exists(extracted_audio_path):
        os.remove(extracted_audio_path)

    # Spleeter 결과 폴더 삭제 (예: extracted_audio/{base_name})
    spleeter_folder = os.path.join("extracted_audio", base_name)
    if os.path.exists(spleeter_folder):
        shutil.rmtree(spleeter_folder)

    # 관련 DB 레코드 삭제 (예: transcripts, translations, TTS, background_music, videos)
    try:
        # TTS 삭제: transcripts와 연결된 translation, tts 기록 삭제
        curs.execute("""
            DELETE FROM tts 
            WHERE translation_id IN (
                SELECT translation_id FROM translations 
                WHERE transcript_id IN (
                    SELECT transcript_id FROM transcripts WHERE video_id = %s
                )
            );
        """, (video_id,))
        curs.execute("""
            DELETE FROM translations 
            WHERE transcript_id IN (
                SELECT transcript_id FROM transcripts WHERE video_id = %s
            );
        """, (video_id,))
        curs.execute("DELETE FROM transcripts WHERE video_id = %s;", (video_id,))
        curs.execute("DELETE FROM background_music WHERE video_id = %s;", (video_id,))
        curs.execute("DELETE FROM videos WHERE video_id = %s;", (video_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="DB 삭제 중 오류 발생")
    finally:
        curs.close()
        conn.close()

    return {"detail": "영상 및 관련 파일이 성공적으로 삭제되었습니다."}

@app.delete("/delete-audio-file")
async def delete_audio_file(
    file: str = Query(..., description="삭제할 오디오 파일명"),
    folder: str = Query(..., description="파일이 속한 폴더 (custom_tts 또는 sound_effects)")
):
    allowed_folders = ["custom_tts", "sound_effects"]
    if folder not in allowed_folders:
        raise HTTPException(status_code=400, detail="허용되지 않은 폴더입니다.")
    
    file_path = os.path.join("extracted_audio", folder, file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    try:
        os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 삭제 중 오류 발생: {str(e)}")
    
    return {"detail": f"{folder} 폴더의 {file} 파일이 삭제되었습니다."}

@app.get("/videos/{video_id}/subtitles")
async def get_video_and_subtitles(video_id: int, request: Request):
    """
    video_id에 해당하는 비디오의 상대경로 URL(예: '/uploaded_videos/{file_name}')과,
    transcripts/translation을 합쳐서 만든 자막 리스트를 반환합니다.
    """
    # 1) 인증 검사
    user_id = get_current_user_id(request)

    conn = None
    try:
        conn = get_connection()
        curs = conn.cursor()

        # 2) videos 테이블에서 비디오 정보 조회
        curs.execute(
            "SELECT file_name, file_path, duration FROM videos WHERE video_id = %s;",
            (video_id,)
        )
        video_row = curs.fetchone()
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")

        file_name, file_path, duration = video_row
        # 상대경로로만 내려줌 (/uploaded_videos/{file_name})
        relative_video_path = f"/uploaded_videos/{file_name}"

        # 3) transcripts + translations(있으면) 조인해서 자막 조회
        query = """
            SELECT
                t.transcript_id   AS id,
                t.start_time,
                t.end_time,
                COALESCE(tr.text, t.text) AS text
            FROM transcripts t
            LEFT JOIN translations tr
              ON t.transcript_id = tr.transcript_id
            WHERE t.video_id = %s
            ORDER BY t.start_time ASC;
        """
        curs.execute(query, (video_id,))
        rows = curs.fetchall()
        curs.close()
        conn.close()

        # 4) JSON으로 매핑
        subtitles = []
        for row in rows:
            subtitles.append({
                "id": int(row[0]),
                "start_time": float(row[1]),
                "end_time": float(row[2]),
                "text": row[3] or ""
            })

        # 5) 최종 반환: 상대경로 video URL + 자막 배열
        return JSONResponse(
            content={
                "video": {
                    "video_id": video_id,
                    "file_name": file_name,
                    "relative_url": relative_video_path,
                    "duration": float(duration)
                },
                "subtitles": subtitles
            },
            status_code=200
        )

    except HTTPException as he:
        if conn:
            conn.close()
        raise he

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {e}")