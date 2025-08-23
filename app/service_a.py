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
import logging
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

import math
import subprocess, shlex

# 맨 위 import 구역에 추가
import numpy as np
from PIL import Image, ImageDraw
import librosa
from uuid import uuid4
from pathlib import Path

from urllib.parse import urlparse, unquote
import re

logging.basicConfig(level=logging.DEBUG)

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

ASSEMBLYAI_API_KEY = "assemble-key"

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


# 폴더 생성 섹션 근처에 추가
WAVEFORM_FOLDER = "waveforms"
os.makedirs(WAVEFORM_FOLDER, exist_ok=True)

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

# 정적 파일 제공 (영상 파일과 추출된 오디오 파일)
app.mount("/uploaded_videos", StaticFiles(directory="uploaded_videos"), name="videos")
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")
app.mount("/user_files", StaticFiles(directory="user_files"), name="user_files")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")
app.mount("/waveforms", StaticFiles(directory=WAVEFORM_FOLDER), name="waveforms")

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

def _run(cmd: str):
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stderr}")
    return proc.stdout

def probe_duration(path: str) -> float:
    out = run_cmd_text_out(
        f'ffprobe -v error -show_entries format=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{path}"'
    )
    return float(out.strip())

def make_sprite(input_path: str, out_path: str, thumb_height: int = 100):
    duration = probe_duration(input_path)
    tiles = max(1, math.ceil(duration))
    vf = f'fps=1,scale=-1:{thumb_height},tile={tiles}x1'
    run_cmd_no_stdout(f'ffmpeg -y -i "{input_path}" -vf "{vf}" -frames:v 1 "{out_path}"')
    return duration, tiles


# --- subprocess safe wrappers ---------------------------------
def run_cmd_no_stdout(cmd: str) -> None:
    """
    파일→파일 변환처럼 stdout이 필요 없을 때.
    stderr만 UTF-8로 받아서 에러 메시지 확보.
    """
    proc = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.DEVNULL,                 # ✅ stdout 비활성(바이너리 섞임 차단)
        stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='replace'  # ✅ cp949 회피
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f'cmd failed: {cmd}')

def run_cmd_text_out(cmd: str) -> str:
    """
    ffprobe처럼 stdout '텍스트' 결과가 필요한 경우.
    """
    proc = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,                    # ✅ 텍스트 결과 받기
        stderr=subprocess.PIPE,                    # 로그도 텍스트
        text=True, encoding='utf-8', errors='replace'
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f'cmd failed: {cmd}')
    return proc.stdout
# ---------------------------------------------------------------

# ===== Diarization 기본 힌트(언어별) =====
DIARIZATION_DEFAULTS = {
    "ko": {"min": 1, "max": 3},
    "en": {"min": 1, "max": 3},
    "ja": {"min": 1, "max": 3},
    "zh": {"min": 1, "max": 3},
}


def _norm_lang_key(lang: str) -> str:
    """ 'ko-KR','en-US','ja-JP','zh-CN' → ko/en/ja/zh 로 정규화 """
    if not lang:
        return "ko"
    l = lang.lower()
    if l.startswith("ko"): return "ko"
    if l.startswith("en"): return "en"
    if l.startswith("ja"): return "ja"
    if l.startswith("zh"): return "zh"
    return "ko"

def preprocess_audio_for_stt(src_path: str) -> str:
    """
    STT 안정화를 위한 경량 전처리:
    - mono, 16kHz
    - 기본 필터(고역/저역 + dynaudnorm)
    결과를 임시 wav로 반환
    """
    out_path = os.path.join(tempfile.gettempdir(), f"stt_{uuid4().hex}.wav")
    # ffmpeg 빌드에 dynaudnorm 포함(기본 full build엔 포함됨)
    # 필터는 과하지 않게: 고역 120Hz, 저역 3800Hz, 다이나믹 노멀라이즈
    cmd = (
        f'ffmpeg -y -i "{src_path}" -ac 1 -ar 16000 -vn '
        f'-af "highpass=f=120,lowpass=f=3800,dynaudnorm=f=200:g=15" '
        f'"{out_path}"'
    )
    run_cmd_no_stdout(cmd)
    return out_path

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
    curs.execute("""
        SELECT p.project_id,
               p.project_name,
               p.description,
               p.created_at,
               MIN(v.video_id) AS first_video_id
          FROM projects p
          LEFT JOIN videos v ON v.project_id = p.project_id
         WHERE p.user_id = %s
         GROUP BY p.project_id, p.project_name, p.description, p.created_at
         ORDER BY p.created_at DESC;
    """, (user_id,))
    rows = curs.fetchall()
    curs.close()
    conn.close()

    projects = []
    for project_id, name, desc, created_at, first_video_id in rows:
        # cover 우선, 없으면 sprite
        thumb_cover_rel = f"/thumbnails/{first_video_id}-cover.jpg" if first_video_id else None
        thumb_sprite_rel = f"/thumbnails/{first_video_id}.png"       if first_video_id else None

        cover_abs = os.path.join(THUMBNAIL_FOLDER, f"{first_video_id}-cover.jpg") if first_video_id else None
        sprite_abs = os.path.join(THUMBNAIL_FOLDER, f"{first_video_id}.png")       if first_video_id else None

        if first_video_id and cover_abs and os.path.exists(cover_abs):
            thumbnail_url = thumb_cover_rel
        elif first_video_id and sprite_abs and os.path.exists(sprite_abs):
            thumbnail_url = thumb_sprite_rel
        else:
            thumbnail_url = None

        projects.append({
            "project_id": project_id,
            "project_name": name,
            "description": desc,
            "created_at": created_at.isoformat() if created_at else None,
            "video_id": first_video_id,
            "thumbnail_url": thumbnail_url,
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
            placeholder = "static/audio-placeholder.png"
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

# 2. 사용자별 파일 목록 조회 (디스크 + DB sound_effects 병합)
@app.get("/user-files")
async def list_user_files(request: Request):
    user_id = get_user_id_from_token(request)  # 기존 로그인 방식
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))

    items = []

    # ── 디스크 파일 수집
    if os.path.exists(user_folder):
        for name in os.listdir(user_folder):
            fp = os.path.join(user_folder, name)
            if not os.path.isfile(fp):
                continue
            base, _ = os.path.splitext(name)
            thumb_path = os.path.join(THUMBNAIL_FOLDER, f"{base}_thumbnail.jpg")
            thumb_url = f"/thumbnails/{base}_thumbnail.jpg" if os.path.exists(thumb_path) else "/thumbnails/audio-placeholder.png"
            items.append({
                "file_name": name,
                "file_url": f"/user_files/{user_id}/{name}",
                "thumbnail_url": thumb_url,
                "_source": "disk",
                "_mtime": os.path.getmtime(fp),
            })

    # ── DB 프리셋 수집
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("""
            SELECT name, description, file_path, thumbnail_url, duration, created_at
              FROM sound_effects
             WHERE user_id = %s
             ORDER BY created_at DESC
        """, (user_id,))
        rows = curs.fetchall()
        for name, desc, file_path, thumb, duration, created_at in rows:
            file_url = file_path
            file_name = os.path.basename(file_path) if file_path else (name or "sound.mp3")
            items.append({
                "file_name": file_name,
                "file_url": file_url,
                # DB에서 가져온 썸네일 사용, 없으면 placeholder
                "thumbnail_url": thumb if thumb else "/thumbnails/audio-placeholder.png",
                "duration": float(duration) if duration is not None else None,
                "_source": "db",
                "_ctime": created_at.timestamp() if created_at else 0,
            })
    except Exception as e:
        logging.error(f"DB 조회 실패: {e}")
    finally:
        try:
            curs.close()
            conn.close()
        except:
            pass

    # ── 중복 제거 (DB 우선)
    merged = {}
    for it in items:
        key = it["file_url"]
        if key not in merged or it.get("_source") == "db":
            merged[key] = it

    files = list(merged.values())
    files.sort(key=lambda x: (x.get("_ctime", 0), x.get("_mtime", 0)), reverse=True)

    for f in files:
        f.pop("_source", None)
        f.pop("_ctime", None)
        f.pop("_mtime", None)

    return {"files": files}

# 3. 사용자별 파일 삭제 (디스크 + DB sound_effects 정리)
@app.delete("/user-files")
async def delete_user_file(request: Request, file: str = Query(...)):
    user_id = get_user_id_from_token(request)
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    file_path = os.path.join(user_folder, file)

    # ── 디스크 파일 삭제 (있을 때만)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {e}")

        # 썸네일 삭제(있으면)
        base_name, _ = os.path.splitext(file)
        thumb_path = os.path.join(THUMBNAIL_FOLDER, f"{base_name}_thumbnail.jpg")
        if os.path.exists(thumb_path):
            try:
                os.remove(thumb_path)
            except:
                pass

    # ── DB 프리셋 레코드 삭제(있으면)
    file_url = f"/user_files/{user_id}/{file}"
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "DELETE FROM sound_effects WHERE user_id = %s AND file_path = %s",
            (user_id, file_url)
        )
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            curs.close(); conn.close()
        except:
            pass

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
def clova_speech_stt(
    file_path: str,
    completion: str = "sync",
    language: str = "ko-KR",
    wordAlignment: bool = True,
    fullText: bool = True,
    speakerCountMin: int | None = None,
    speakerCountMax: int | None = None,
) -> dict:
    """
    Clova Speech Long Sentence API 호출.
    - speakerCountMin/Max를 넘기지 않으면 언어별 기본값(1~3명)을 적용.
    - 넘기면 전달값을 우선 사용.
    """
    # 기본값 보정
    key = _norm_lang_key(language)
    dflt = DIARIZATION_DEFAULTS.get(key, {"min": 1, "max": 3})
    spk_min = speakerCountMin if speakerCountMin is not None else dflt["min"]
    spk_max = speakerCountMax if speakerCountMax is not None else dflt["max"]
    if spk_min < 1: spk_min = 1
    if spk_max < spk_min: spk_max = spk_min

    diarization = {
        "enable": True,
        "speakerCountMin": int(spk_min),
        "speakerCountMax": int(spk_max),
    }

    request_body = {
        "language": language,
        "completion": completion,
        "wordAlignment": wordAlignment,
        "fullText": fullText,
        "diarization": diarization,
    }

    headers = {
        "Accept": "application/json; charset=UTF-8",
        "X-CLOVASPEECH-API-KEY": NAVER_CLOVA_SECRET_KEY,
    }

    files = {
        "media": open(file_path, "rb"),
        "params": (None, json.dumps(request_body).encode("UTF-8"), "application/json"),
    }
    try:
        url = NAVER_CLOVA_SPEECH_URL + "/recognizer/upload"
        response = requests.post(url, headers=headers, files=files, timeout=120)
    finally:
        files["media"].close()

    if response.status_code != 200:
        print("Clova Speech API 실패:", response.status_code, response.text)
        return {}
    return response.json()

############################################
# AssemblyAI 다이아리제이션 + 트랜스크립션
############################################
def call_assemblyai_with_diarization(audio_path: str) -> dict:
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    # 1) 파일 업로드
    with open(audio_path, "rb") as f:
        upload_resp = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": ASSEMBLYAI_API_KEY},
            data=f
        )
    upload_resp.raise_for_status()
    upload_url = upload_resp.json()["upload_url"]

    # 2) 트랜스크립션 + 스피커 라벨 요청
    transcript_req = {
        "audio_url": upload_url,
        "speaker_labels": True
    }
    transcript_resp = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=transcript_req,
        headers=headers
    )
    transcript_resp.raise_for_status()
    transcript_id = transcript_resp.json()["id"]

    # 3) 완료될 때까지 폴링
    while True:
        status_resp = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers=headers
        )
        status_resp.raise_for_status()
        result = status_resp.json()
        if result["status"] == "completed":
            return result
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        time.sleep(3)


############################################
# STT 변환 함수 (모든 언어 → Clova + diarization)
############################################
async def transcribe_audio(
    audio_path: str,
    video_id: int,
    source_language: str,
    speaker_min: int | None = None,
    speaker_max: int | None = None,
):
    """
    모든 언어(ko/en/ja/zh …)를 Clova Speech Long Sentence API로 처리합니다.
    - diarization 기본값은 DIARIZATION_DEFAULTS의 언어별 min/max(기본 1~3명)를 사용
    - speaker_min/max를 인자로 넘기면 해당 값이 우선됩니다.
    """
    start_time = time.time()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 언어별 화자수 기본 적용 or 오버라이드
    if speaker_min is None or speaker_max is None:
        key = _norm_lang_key(source_language)
        dflt = DIARIZATION_DEFAULTS.get(key, {"min": 1, "max": 3})
        speaker_min = dflt["min"] if speaker_min is None else speaker_min
        speaker_max = dflt["max"] if speaker_max is None else speaker_max

    # (선택) STT 안정화 전처리해서 넣고 싶다면 아래 두 줄로 교체:
    # prepped = preprocess_audio_for_stt(audio_path)
    # stt_src = prepped
    stt_src = audio_path

    conn = get_connection()
    curs = conn.cursor()

    try:
        result = clova_speech_stt(
            stt_src,
            language=source_language,
            speakerCountMin=speaker_min,
            speakerCountMax=speaker_max,
        )

        # Clova 응답 파싱
        for seg in result.get("segments", []):
            s = float(seg.get("start", 0)) / 1000.0
            e = float(seg.get("end",   0)) / 1000.0
            txt = seg.get("text", "")
            spk_info = seg.get("speaker", {}) or {}
            spk = spk_info.get("name") or spk_info.get("label") or ""
            curs.execute(
                """
                INSERT INTO transcripts
                  (video_id, language, text, start_time, end_time, speaker)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (video_id, source_language, txt, s, e, spk),
            )

        conn.commit()
        return {"stt_time": time.time() - start_time}

    except Exception:
        conn.rollback()
        raise
    finally:
        curs.close()
        conn.close()


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

    duration = float(video[3])
    video_data = {
        "video_id": video[0],
        "file_name": video[1],
        "file_path": video[2],
        "duration": duration,
        "thumbnail_url": f"/thumbnails/{video[0]}.png",
        "width_px": int(math.ceil(duration) * 100),
    }

    curs.execute("SELECT file_path, volume, waveform_url FROM background_music WHERE video_id=%s;", (video_id,))
    bgm = curs.fetchone()
    background_music = {
        "file_path": bgm[0] if bgm else None,
        "volume": float(bgm[1]) if bgm else 1.0,
        "waveform_url": bgm[2] if bgm else None,
    }

    curs.execute(
        """
        SELECT t.tts_id, t.file_path, t.voice, t.start_time, t.duration, 
            tr.text as translated_text, ts.text as original_text, ts.speaker,
            t.waveform_url
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
            "speaker": row[7],
            "waveform_url": row[8],
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

        # 4. DB 저장 (videos + background_music)
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

        # 4-1. 썸네일(스프라이트 PNG) 생성
        thumb_dir = THUMBNAIL_FOLDER
        os.makedirs(thumb_dir, exist_ok=True)
        sprite_path = os.path.join(thumb_dir, f"{video_id}.png")
        duration, tiles = make_sprite(file_path, sprite_path, thumb_height=100)
        width_px = int(math.ceil(duration) * 100)

        # 4-2. 커버 JPG 생성 (첫 프레임)
        cover_path = os.path.join(thumb_dir, f"{video_id}-cover.jpg")
        try:
            with VideoFileClip(file_path) as cover_clip:
                cover_clip.save_frame(cover_path, t=0.0)
        except Exception:
            try:
                with VideoFileClip(file_path) as cover_clip:
                    cover_clip.save_frame(cover_path, t=max(0.0, duration/2))
            except Exception as e:
                logging.warning(f"cover thumbnail failed: {e}")

        # 4-3. 🔹 BGM 파형 PNG 생성 + DB 반영
        try:
            bgm_path = separation_data.get('bgm_path')  # Spleeter 결과의 accompaniment 경로
            if bgm_path and os.path.exists(bgm_path):
                wave_name = f"bgm_{video_id}.png"
                wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
                # 타임라인 규칙: 1초 = 100px
                generate_waveform_png(bgm_path, wave_path, width_px=width_px, height_px=100)

                def update_bgm_waveform(vid, url):
                    conn = get_connection(); curs = conn.cursor()
                    curs.execute(
                        "UPDATE background_music SET waveform_url=%s WHERE video_id=%s;",
                        (url, vid)
                    )
                    conn.commit(); curs.close(); conn.close()

                await to_thread(update_bgm_waveform, video_id, f"/waveforms/{wave_name}")
        except Exception as e:
            logging.warning(f"BGM waveform generation failed: {e}")

        # 5. STT
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

        # 7. TTS 생성 (TTS 파형은 service_b에서 생성/저장)
        await to_thread(
            lambda vid: requests.post(f"{API_HOST}/generate-tts-from-stt", json={'video_id': vid}).raise_for_status(),
            video_id
        )
        await ws_manager.send_progress(job_id, cumulative_pct['tts_time'])
        await asyncio.sleep(1)

        # 8. 최종 데이터 조회
        _ = await get_edit_data(video_id)
        await ws_manager.send_progress(job_id, cumulative_pct['get_time'])

        return JSONResponse(content={'job_id': job_id}, status_code=202)

    except Exception as e:
        await ws_manager.send_progress(job_id, -1)
        raise HTTPException(status_code=500, detail=f"업로드 실패: {e}")

def resolve_local_path(url: str) -> str:
    parsed = urlparse(url)
    host = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname

    # ✅ URL 경로를 디코드 + // 정리
    p = unquote((parsed.path or "/")).replace("//", "/")

    if host in LOCAL_HOSTS:
        for prefix, dirpath in STATIC_MAP.items():
            if p.startswith(prefix) or p.startswith("/" + prefix.lstrip("/")):
                use_prefix = prefix if p.startswith(prefix) else ("/" + prefix.lstrip("/"))
                # ✅ 상대 경로도 디코드
                rel = unquote(p[len(use_prefix):]).lstrip("/")
                return os.path.join(dirpath, rel)

    # ✅ host 없는 상대경로도 처리 (예: "/uploaded_videos/..")
    if not host:
        for prefix, dirpath in STATIC_MAP.items():
            if p.startswith(prefix) or p.startswith("/" + prefix.lstrip("/")):
                use_prefix = prefix if p.startswith(prefix) else ("/" + prefix.lstrip("/"))
                rel = unquote(p[len(use_prefix):]).lstrip("/")
                return os.path.join(dirpath, rel)

    return url


# --- ASS subtitle helpers ------------------------------------

def _ass_escape(text: str) -> str:
    # ASS 제어문자 이스케이프
    return (text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

def _ff_color_to_ass(color: str) -> str:
    """
    #RRGGBB -> &HBBGGRR& (ASS BGR)
    """
    if not color or not color.startswith("#") or len(color) != 7:
        return "&HFFFFFF&"  # 기본 흰색
    r = color[1:3]; g = color[3:5]; b = color[5:7]
    return f"&H{b}{g}{r}&"

def _sec_to_ass_time(t: float) -> str:
    # 0.00s -> 0:00:00.00
    if t < 0: t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"

def write_ass(subs: list, ass_path: str, width: int = 1280, height: int = 720, style: dict | None = None):
    """
    subs: [{text,start,end,color}]
    style: {
      "fontName": "Pretendard",
      "fontScale": 0.056,  # 기본값(2배). 화면 짧은 변 × 비율 → pt 환산
      "fontSize": 64,      # 명시하면 fontScale 무시
      "outline": 3,
      "shadow": 0,
      "alignment": 2,      # 2=하단 중앙
      "marginV": 28
    }
    """
    style = style or {}

    # --- 폰트 크기 계산(기본 2배) ---
    short_side = min(width, height)
    font_scale = float(style.get("fontScale", 0.056))  # ← 2배 기본
    font_size  = int(round(style.get("fontSize", short_side * font_scale))) \
                 if "fontSize" in style else int(round(short_side * font_scale))

    font_name = style.get("fontName", "Pretendard")
    outline   = int(style.get("outline", 3))
    shadow    = int(style.get("shadow", 0))
    align     = int(style.get("alignment", 2))
    margin_v  = int(style.get("marginV", 28))

    def _esc(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    def _ff_to_ass_bgr(hexrgb: str) -> str:
        if not hexrgb or not hexrgb.startswith("#") or len(hexrgb) != 7:
            return "&HFFFFFF&"
        r, g, b = hexrgb[1:3], hexrgb[3:5], hexrgb[5:7]
        return f"&H{b}{g}{r}&"

    def _sec(t: float) -> str:
        if t < 0: t = 0
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return f"{h:d}:{m:02d}:{s:05.2f}"

    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&HDD000000,&H99000000,0,0,0,0,100,100,0,0,3,{outline},{shadow},{align},60,60,{margin_v},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]

    for item in subs:
        text = _esc(str(item.get("text", "")))
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start + 1.0))
        if end <= start:
            end = start + 1.0

        cue_color = _ff_to_ass_bgr(item.get("color", "#FFFFFF"))
        ass_text = f"{{\\c{cue_color}}}▌{{\\c&H00FFFFFF&}} {text} {{\\c{cue_color}}}▌"

        lines.append(
            f"Dialogue: 0,{_sec(start)},{_sec(end)},Default,,0,0,0,,{ass_text}\n"
        )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

@app.post("/merge-media")
async def merge_media(request: Request):
    """
    payload 예시
    {
      "videoTracks": [...],
      "audioTracks": [...],
      "subtitles": [...],
      "subtitlesStyle": {"fontScale": 0.024, "outline": 3, "marginV": 36},
      "upscale": 2.0,                           // ← 배율(옵션)
      "targetResolution": {"width": 2560, "height": 1440} // ← 우선순위 더 높음(옵션)
    }
    """
    payload = await request.json()
    tempdir = tempfile.mkdtemp(prefix="merge_")

    subtitles   = payload.get("subtitles", []) or []
    subs_style  = payload.get("subtitlesStyle", {}) or {}
    upscale     = float(payload.get("upscale", 1.0))
    tgt_res_in  = payload.get("targetResolution") or {}

    video_items = []  # (priority, clip)
    audio_clips = []

    def fetch_and_resolve(raw_url: str) -> str:
        logging.info(f"[merge] raw_url={raw_url}")
        path_or_url = resolve_local_path(raw_url)
        logging.info(f"[merge] resolved={path_or_url}")

        if isinstance(path_or_url, str) and not path_or_url.startswith("http"):
            path_or_url = unquote(path_or_url)
            if not os.path.isfile(path_or_url):
                raise HTTPException(404, f"파일을 찾을 수 없습니다: {raw_url} -> {path_or_url}")
            return path_or_url

        r = requests.get(path_or_url, stream=True, timeout=300)
        r.raise_for_status()
        ext = os.path.splitext(urlparse(path_or_url).path)[1] or ".bin"
        fn = os.path.join(tempdir, f"dl_{time.time():.0f}{ext}")
        with open(fn, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        return fn

    # ----- 비디오 트랙 -----
    for group in payload.get("videoTracks", []) or []:
        try:
            priority = int(group.get("name", "").strip().split()[-1])
        except Exception:
            priority = 0
        for track in group.get("tracks", []) or []:
            url = track.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="video url 누락")
            start = float(track.get("startTime", 0))
            fp = fetch_and_resolve(url)
            vclip = VideoFileClip(fp).set_start(start).without_audio()
            video_items.append((priority, vclip))

    # ----- 오디오 트랙 -----
    for group in payload.get("audioTracks", []) or []:
        vol = float(group.get("volume", 100)) / 100.0
        for track in group.get("tracks", []) or []:
            url = track.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="audio url 누락")
            start = float(track.get("startTime", 0))
            fp = fetch_and_resolve(url)
            aclip = AudioFileClip(fp).set_start(start).volumex(vol)
            audio_clips.append(aclip)

    if not video_items and not audio_clips:
        raise HTTPException(status_code=400, detail="합성할 트랙이 없습니다.")

    # 전경 우선
    video_items.sort(key=lambda x: -x[0])
    ordered_vclips = [c for _, c in video_items]
    if not ordered_vclips:
        raise HTTPException(status_code=400, detail="비디오 트랙이 최소 1개 필요합니다.")

    base_w, base_h = ordered_vclips[0].size
    try:
        target_fps = int(round(ordered_vclips[0].fps)) if getattr(ordered_vclips[0], "fps", None) else None
    except Exception:
        target_fps = None

    final_video = CompositeVideoClip(ordered_vclips, size=(base_w, base_h))
    if audio_clips:
        final_video.audio = CompositeAudioClip(audio_clips)
    else:
        final_video = final_video.set_audio(None)

    # 1차 출력(메자닌) — 고품질 인코딩
    mezz_path = os.path.join(tempdir, "mezzanine.mp4")
    write_kwargs = {
        "codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "preset": "slow",
        "ffmpeg_params": [
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "high",
            "-level", "4.1"
        ]
    }
    if target_fps:
        write_kwargs["fps"] = target_fps
    final_video.write_videofile(mezz_path, **write_kwargs)

    # 리소스 해제
    for _, c in video_items:
        try: c.close()
        except: pass
    for a in audio_clips:
        try: a.close()
        except: pass

    # 최종 해상도 결정: targetResolution > upscale > 원본
    if isinstance(tgt_res_in, dict) and tgt_res_in.get("width") and tgt_res_in.get("height"):
        final_w = int(tgt_res_in["width"])
        final_h = int(tgt_res_in["height"])
    else:
        # upscale 배율 적용 (짝수 보장: H.264 요건)
        final_w = int(round(base_w * max(1.0, upscale)) // 2 * 2)
        final_h = int(round(base_h * max(1.0, upscale)) // 2 * 2)

    need_scale = (final_w != base_w) or (final_h != base_h)

    # 자막: 최종 해상도 기준으로 작성 (폰트 비례)
    final_path = mezz_path
    if subtitles:
        ass_path = os.path.join(tempdir, "burn.ass")
        write_ass(subtitles, ass_path, width=final_w, height=final_h, style=subs_style)

        burned_path = os.path.join(tempdir, "merged_final.mp4")
        ass_arg = ass_path.replace("\\", "/")
        if len(ass_arg) >= 2 and ass_arg[1] == ":":
            ass_arg = ass_arg[0] + "\\:" + ass_arg[2:]

        # 스케일 → 자막 번인 순서로 필터 체인 구성
        vf_chain = []
        if need_scale:
            vf_chain.append(f"scale={final_w}:{final_h}:flags=lanczos")
            # (선택) 가장자리 선명도 약간 보정
            # vf_chain.append("unsharp=3:3:0.3:3:3:0.3")
        vf_chain.append(f"subtitles=filename='{ass_arg}'")
        vf = ",".join(vf_chain)

        # 업스케일은 화질 손실이 크게 느껴지므로 CRF를 더 낮춰(=고화질) 인코딩 권장
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{mezz_path}" -vf "{vf}" '
            f'-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p '
            f'-profile:v high -level 4.1 -movflags +faststart '
            f'-c:a aac -b:a 192k '
        )
        if target_fps:
            ffmpeg_cmd += f'-r {target_fps} '
        ffmpeg_cmd += f'"{burned_path}"'

        run_cmd_no_stdout(ffmpeg_cmd)
        final_path = burned_path

    else:
        # 자막이 없으면 스케일만 적용 (필요 시)
        if need_scale:
            scaled_path = os.path.join(tempdir, "merged_final.mp4")
            vf = f"scale={final_w}:{final_h}:flags=lanczos"
            ffmpeg_cmd = (
                f'ffmpeg -y -i "{mezz_path}" -vf "{vf}" '
                f'-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p '
                f'-profile:v high -level 4.1 -movflags +faststart '
                f'-c:a aac -b:a 192k '
            )
            if target_fps:
                ffmpeg_cmd += f'-r {target_fps} '
            ffmpeg_cmd += f'"{scaled_path}"'
            run_cmd_no_stdout(ffmpeg_cmd)
            final_path = scaled_path

    return FileResponse(
        path=final_path,
        filename=os.path.basename(final_path),
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
    conn = get_connection()
    curs = conn.cursor()
    curs.execute(
        "SELECT video_id, file_name, file_path, duration FROM videos WHERE file_name = %s ORDER BY video_id DESC LIMIT 1;",
        (filename,)
    )
    video = curs.fetchone()
    if not video:
        raise HTTPException(status_code=404, detail="해당 영상이 존재하지 않습니다.")
    video_id, file_name, file_path, duration = video
    base_name = os.path.splitext(filename)[0]

    # 썸네일 삭제
    sprite = os.path.join("thumbnails", f"{video_id}.png")
    cover  = os.path.join("thumbnails", f"{video_id}-cover.jpg")
    for p in (sprite, cover):
        if os.path.exists(p):
            try: os.remove(p)
            except: pass

    # 영상 파일 삭제
    if os.path.exists(file_path):
        os.remove(file_path)

    # 추출된 오디오 파일 삭제
    extracted_audio_path = os.path.join("extracted_audio", f"{base_name}.mp3")
    if os.path.exists(extracted_audio_path):
        os.remove(extracted_audio_path)

    # Spleeter 폴더 삭제
    spleeter_folder = os.path.join("extracted_audio", base_name)
    if os.path.exists(spleeter_folder):
        shutil.rmtree(spleeter_folder)

    # DB 삭제
    try:
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
    except Exception:
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
    
def generate_waveform_png(audio_path: str, out_path: str, width_px: int, height_px: int = 100, bg="#FFFFFF", fg="#007bff", sr: int = 22050):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    n = len(y) or sr // 2
    if len(y) == 0:
        import numpy as _np
        y = _np.zeros(sr // 2)

    samples_per_px = max(1, len(y) // width_px)
    peaks = []
    for i in range(width_px):
        s = i * samples_per_px
        e = min(len(y), s + samples_per_px)
        amp = float(np.max(np.abs(y[s:e]))) if s < len(y) else 0.0
        peaks.append(amp)
    maxamp = max(peaks) or 1.0
    peaks = [p / maxamp for p in peaks]

    img = Image.new("RGB", (width_px, height_px), bg)
    draw = ImageDraw.Draw(img)
    mid = height_px / 2.0
    for x, p in enumerate(peaks):
        h = p * (height_px * 0.9)
        y0 = int(mid - h/2); y1 = int(mid + h/2)
        draw.line((x, y0, x, y1), fill=fg, width=1)
    img.save(out_path, format="PNG")


@app.post("/audio/upload")
async def upload_audio_and_waveform(file: UploadFile = File(...)):
    """
    오디오 파일 저장 + 파형 PNG 생성
    반환: { audio_url, waveform_url, duration, width_px }
    """
    try:
        # 저장 위치: extracted_audio/sound_effects/*
        ext = os.path.splitext(file.filename)[1].lower() or ".mp3"
        audio_name = f"{uuid4().hex}{ext}"
        audio_path = os.path.join(AUDIO_FOLDER, "sound_effects", audio_name)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # 길이 → 타임라인 폭(px)
        duration = float(librosa.get_duration(path=audio_path))
        width_px = int(math.ceil(duration * 100))  # 규칙: 1초 = 100px

        # 파형 PNG
        wave_name = f"{os.path.splitext(audio_name)[0]}.png"
        wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
        generate_waveform_png(audio_path, wave_path, width_px=width_px, height_px=100)

        return JSONResponse(
            {
                "audio_url": f"/extracted_audio/sound_effects/{audio_name}",
                "waveform_url": f"/waveforms/{wave_name}",
                "duration": duration,
                "width_px": width_px
            },
            status_code=201
        )
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=f"파형 생성 실패: {e}")


@app.post("/audio/from-url")
async def audio_from_url_and_waveform(src_url: str = Form(...)):
    """
    외부 오디오 URL 다운로드 + 파형 PNG 생성
    반환: { audio_url, waveform_url, duration, width_px }
    """
    try:
        r = requests.get(src_url, stream=True, timeout=30)
        r.raise_for_status()

        parsed_ext = os.path.splitext(src_url.split("?")[0])[1].lower()
        ext = parsed_ext if parsed_ext in [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"] else ".mp3"

        audio_name = f"{uuid4().hex}{ext}"
        audio_path = os.path.join(AUDIO_FOLDER, "sound_effects", audio_name)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        with open(audio_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        duration = float(librosa.get_duration(path=audio_path))
        width_px = int(math.ceil(duration * 100))  # 1초 = 100px

        wave_name = f"{os.path.splitext(audio_name)[0]}.png"
        wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
        generate_waveform_png(audio_path, wave_path, width_px=width_px, height_px=100)

        return JSONResponse(
            {
                "audio_url": f"/extracted_audio/sound_effects/{audio_name}",
                "waveform_url": f"/waveforms/{wave_name}",
                "duration": duration,
                "width_px": width_px
            },
            status_code=201
        )
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=f"URL 파형 생성 실패: {e}")
    