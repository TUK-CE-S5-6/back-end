import os
import time
import json
import openai
import psycopg2
import requests
import asyncio
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Response, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
import moviepy.video.fx.all as vfx

BASE_HOST = "http://localhost"  # 또는 배포 시 EC2 인스턴스의 공인 IP나 도메인

API_HOST = f"{BASE_HOST}:8001"
SPLITTER_HOST = f"{BASE_HOST}:8001"

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

# 사용자별 파일 관리를 위한 폴더 경로 설정
USER_FILES_FOLDER = "user_files"
os.makedirs(USER_FILES_FOLDER, exist_ok=True)

# Pydantic 모델 (사용자 관련 예시)
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    user_id: int
    username: str


############################################
# 회원가입 엔드포인트 (토큰을 쿠키에 설정)
############################################
@app.post("/signup")
async def signup(response: Response, username: str = Form(...), password: str = Form(...)):
    try:
        conn = get_connection()
        curs = conn.cursor()
        # 이미 같은 username이 존재하는지 확인
        curs.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        if curs.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        curs.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s) RETURNING user_id",
            (username, password)
        )
        user_id = curs.fetchone()[0]
        conn.commit()
        curs.close()
        conn.close()
        # 예제용 토큰 생성 (실제 서비스에서는 JWT 등 안전한 방식을 사용하세요)
        token = f"token-for-user-{user_id}"
        res = JSONResponse(content={"message": "Signup successful", "user_id": user_id})
        # HttpOnly 쿠키로 토큰 저장: 클라이언트 측 스크립트에서 접근할 수 없으므로 안전함
        res.set_cookie(key="token", value=token, httponly=True, path="/")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

############################################
# 로그인 엔드포인트 (토큰을 쿠키에 설정)
############################################
@app.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute("SELECT user_id, password FROM users WHERE username = %s", (username,))
        result = curs.fetchone()
        if result is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        user_id, stored_password = result
        if password != stored_password:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        # 예제용 토큰. 실제 서비스에서는 JWT 등 안전한 토큰을 사용하세요.
        token = f"token-for-user-{user_id}"
        curs.close()
        conn.close()
        res = JSONResponse(content={"message": "Login successful", "user_id": user_id})
        res.set_cookie(key="token", value=token, httponly=True, path="/")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

############################################
# 현재 로그인한 사용자 정보를 반환하는 엔드포인트
############################################
@app.get("/me")
async def get_current_user(request: Request):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # 예제 토큰의 경우 "token-for-user-{user_id}" 형식이므로, 이를 파싱하여 user_id를 추출
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    return {"user_id": user_id}

############################################
# 사용자별 프로젝트
############################################
@app.get("/projects")
async def get_projects(request: Request):
    # 쿠키에서 토큰 추출 (예제 토큰 형식: "token-for-user-{user_id}")
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    try:
        conn = get_connection()
        curs = conn.cursor()
        # 해당 user_id에 속한 프로젝트를 최신순으로 조회
        curs.execute(
            "SELECT project_id, project_name, description, created_at FROM projects WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        projects = curs.fetchall()
        curs.close()
        conn.close()

        projects_list = [
            {
                "project_id": row[0],
                "project_name": row[1],
                "description": row[2],
                # datetime 객체를 문자열로 변환 (.isoformat())
                "created_at": row[3].isoformat() if row[3] else None
            }
            for row in projects
        ]
        return JSONResponse(content={"projects": projects_list}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/projects/add")
async def add_project(
    request: Request,
    project_name: str = Form(...),
    description: str = Form("")
):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    try:
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "INSERT INTO projects (project_name, description, user_id) VALUES (%s, %s, %s) RETURNING project_id",
            (project_name, description, user_id)
        )
        project_id = curs.fetchone()[0]
        conn.commit()
        curs.close()
        conn.close()
        return JSONResponse(
            content={
                "project_id": project_id,
                "project_name": project_name,
                "description": description
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, request: Request):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        # 예제 토큰 형식: "token-for-user-{user_id}"
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    try:
        conn = get_connection()
        curs = conn.cursor()
        # 현재 사용자 소유의 프로젝트인지 확인
        curs.execute(
            "SELECT project_id FROM projects WHERE project_id = %s AND user_id = %s",
            (project_id, user_id)
        )
        row = curs.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없거나 삭제 권한이 없습니다.")
        
        # 프로젝트 삭제 (참조 무결성은 DB 설정에 따름)
        curs.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))
        conn.commit()
        curs.close()
        conn.close()
        return JSONResponse(content={"message": "프로젝트 삭제 성공"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}/videos/edit_data")
async def get_project_videos_edit_data(project_id: int, request: Request):
    # 쿠키에서 토큰 추출 (예제 토큰 형식: "token-for-user-{user_id}")
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    try:
        conn = get_connection()
        curs = conn.cursor()
        # 해당 프로젝트가 현재 로그인한 사용자의 소유인지 확인
        curs.execute(
            "SELECT project_id FROM projects WHERE project_id = %s AND user_id = %s",
            (project_id, user_id)
        )
        if curs.fetchone() is None:
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없거나 권한이 없습니다.")
        
        # 해당 프로젝트에 속한 모든 영상의 video_id 조회
        curs.execute("SELECT video_id FROM videos WHERE project_id = %s", (project_id,))
        video_ids = [row[0] for row in curs.fetchall()]
        curs.close()
        conn.close()
        
        # 각 video_id에 대해 get_edit_data를 호출하여 상세 정보 수집
        videos_data = []
        for vid in video_ids:
            response = await get_edit_data(vid)
            # get_edit_data는 JSONResponse를 반환하므로, body를 파싱합니다.
            # body가 bytes인 경우 디코딩 후 JSON으로 변환
            if isinstance(response.body, bytes):
                data = json.loads(response.body.decode())
            else:
                data = response.body
            videos_data.append(data)
        
        return JSONResponse(content={"videos": videos_data}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 1. 사용자별 파일 업로드 엔드포인트
@app.post("/upload-file")
async def upload_file(request: Request, file: UploadFile = File(...)):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    # 사용자 전용 폴더 생성 (예: user_files/3)
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse(content={"message": "파일 업로드 성공", "file_path": file_path}, status_code=200)


# 2. 사용자별 파일 목록 조회 엔드포인트
@app.get("/user-files")
async def list_user_files(request: Request):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    if not os.path.exists(user_folder):
        return JSONResponse(content={"files": []}, status_code=200)
    files = [f for f in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, f))]
    return JSONResponse(content={"files": files}, status_code=200)

# 3. 사용자별 파일 삭제 엔드포인트
@app.delete("/user-files")
async def delete_user_file(request: Request, file: str = Query(..., description="삭제할 파일명") ):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")
    user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
    file_path = os.path.join(user_folder, file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    try:
        os.remove(file_path)
        return JSONResponse(content={"message": "파일 삭제 성공"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {str(e)}")

# 3. 사용자별 파일 다운로드 엔드포인트
@app.get("/download-file")
async def download_file(request: Request, file_name: str = Query(...)):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user_id = int(token.split("-")[-1])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")

    # 사용자 전용 폴더
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Translate the following text from {source_language} to {target_language} concisely while ensuring that the translated text's length is as close as possible to the original. Maintain nearly identical word count and sentence structure so that when converted to TTS, the output closely matches the original timing."},
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

############################################
# 영상 업로드 및 처리 엔드포인트
# 클라이언트는 영상 파일과 함께 source_language, target_language(언어 코드)를 폼 데이터로 전달합니다.
############################################
@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    source_language: str = Form("ko-KR"),
    target_language: str = Form("en-US"),
    project_id: int = Form(...),  # 클라이언트에서 선택한 프로젝트 ID를 반드시 전달
):
    overall_start = time.time()
    timings = {}

    try:
        # 1. 영상 파일 저장
        original_file_name = file.filename
        file_name = os.path.splitext(original_file_name)[0]
        base_name = file_name[:-len("_audio")] if file_name.endswith("_audio") else file_name

        step_start = time.time()
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        timings["upload_time"] = time.time() - step_start

        # 2. 영상 길이 계산 및 오디오 추출
        step_start = time.time()
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        extracted_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(extracted_audio_path, codec="mp3")
        audio_clip.close()
        video_clip.close()
        timings["audio_extraction_time"] = time.time() - step_start

        # 3. Spleeter 호출
        step_start = time.time()
        with open(extracted_audio_path, "rb") as audio_file:
            separation_response = requests.post(
                f"{SPLITTER_HOST}/separate-audio",
                files={"file": audio_file}
            )
        if separation_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Spleeter 분리 서비스 호출 실패")
        separation_data = separation_response.json()
        timings["spleeter_time"] = time.time() - step_start

        # 4. DB에 비디오 정보 저장 (project_id 포함)
        step_start = time.time()
        conn = get_connection()
        curs = conn.cursor()
        curs.execute(
            "INSERT INTO videos (file_name, file_path, duration, project_id) VALUES (%s, %s, %s, %s) RETURNING video_id;",
            (file.filename, file_path, duration, project_id)
        )
        video_id = curs.fetchone()[0]
        curs.execute(
            "INSERT INTO background_music (video_id, file_path, volume) VALUES (%s, %s, %s);",
            (video_id, separation_data.get("bgm_path"), 1.0)
        )
        conn.commit()
        curs.close()
        conn.close()
        timings["db_time"] = time.time() - step_start

        # 5. STT 처리
        stt_timings = await transcribe_audio(separation_data.get("vocals_path"), video_id, source_language)
        timings.update(stt_timings)

        # 6. 번역 처리
        translation_timings = await translate_video(video_id, source_language, target_language)
        timings.update(translation_timings)

        # 7. TTS 생성
        step_start = time.time()
        stt_data = {"video_id": video_id}
        tts_response = requests.post(f"{API_HOST}/generate-tts-from-stt", json=stt_data)
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS 생성 서비스 호출 실패")
        timings["tts_time"] = time.time() - step_start

        # 8. 최종 결과 조회 및 영상 합성 (기존 get_edit_data 활용)
        result_response = await get_edit_data(video_id)
        result_data = result_response.body if hasattr(result_response, "body") else {}
        if isinstance(result_data, bytes):
            result_data = json.loads(result_response.body.decode())

        # ---- 오버랩 감지 및 영상 속도 조절 시작 ----
        tts_tracks = result_data.get("tts_tracks", [])
        # 영상 클립 재로딩
        video_clip = VideoFileClip(file_path)
        if len(tts_tracks) >= 2:
            # 정렬: start_time 기준
            tts_tracks_sorted = sorted(tts_tracks, key=lambda x: x["start_time"])
            cumulative_shift = 0  # 누적 시간 변경량
            for i in range(1, len(tts_tracks_sorted)):
                prev = tts_tracks_sorted[i-1]
                curr = tts_tracks_sorted[i]
                prev_end = prev["start_time"] + prev["duration"] + cumulative_shift
                if prev_end > curr["start_time"]:
                    # 겹치는 시간 계산
                    overlap = prev_end - curr["start_time"]
                    original_segment_duration = curr["start_time"] - prev["start_time"]
                    desired_segment_duration = prev["duration"]
                    speed_factor = original_segment_duration / desired_segment_duration
                    new_curr_start = prev["start_time"] + desired_segment_duration
                    shift = new_curr_start - curr["start_time"]
                    # 로그 출력: 원래 구간, 겹침, 속도 인자, shift
                    print(f"Overlap detected between TTS {i} and TTS {i+1}:")
                    print(f"  Previous TTS starts at {prev['start_time']}s with duration {prev['duration']}s (ends at {prev['start_time']+prev['duration']}s)")
                    print(f"  Current TTS original start: {curr['start_time']}s")
                    print(f"  Calculated overlap: {overlap}s")
                    print(f"  Original transcript segment duration: {original_segment_duration}s")
                    print(f"  Desired segment duration (prev TTS duration): {desired_segment_duration}s")
                    print(f"  Speed factor applied: {speed_factor}")
                    print(f"  Shift applied to current TTS: {shift}s, new start time: {curr['start_time'] + shift}s")
                    
                    curr["start_time"] += shift
                    cumulative_shift += shift

                    # 영상 클립 조절: 조절 구간은 [prev.start_time, curr.start_time - shift]
                    clip_before = video_clip.subclip(0, prev["start_time"])
                    clip_overlap = video_clip.subclip(prev["start_time"], curr["start_time"] - shift).fx(vfx.speedx, factor=speed_factor)
                    clip_after = video_clip.subclip(curr["start_time"] - shift, video_clip.duration)
                    video_clip = concatenate_videoclips([clip_before, clip_overlap, clip_after])
        # ---- 오버랩 감지 및 영상 속도 조절 끝 ----

        # 9. 최종 영상 합성: 배경음 및 TTS 트랙 합성
        bgm_path = result_data.get("background_music", {}).get("file_path")
        if not bgm_path:
            raise HTTPException(status_code=500, detail="배경음 파일을 찾을 수 없습니다.")
        background_audio = AudioFileClip(bgm_path)
        
        tts_audio_clips = []
        for tts in result_data.get("tts_tracks", []):
            clip = AudioFileClip(tts["file_path"]).set_start(tts["start_time"])
            tts_audio_clips.append(clip)
        composite_tts_audio = CompositeAudioClip(tts_audio_clips) if tts_audio_clips else None

        audio_clips_to_composite = [background_audio]
        if composite_tts_audio:
            audio_clips_to_composite.append(composite_tts_audio)
        final_audio = CompositeAudioClip(audio_clips_to_composite)
        video_clip.audio = final_audio

        merged_filename = f"merged_{int(time.time())}.mp4"
        merged_output_path = os.path.join(UPLOAD_FOLDER, merged_filename)
        video_clip.write_videofile(merged_output_path, codec="libx264", audio_codec="aac")
        video_clip.close()
        background_audio.close()
        if composite_tts_audio:
            composite_tts_audio.close()
        timings["merge_time"] = time.time() - stt_timings.get("stt_time", 0)

        overall_time = time.time() - overall_start

        result_data["timings"] = {
            "upload_time": timings.get("upload_time", 0),
            "audio_extraction_time": timings.get("audio_extraction_time", 0),
            "spleeter_time": timings.get("spleeter_time", 0),
            "db_time": timings.get("db_time", 0),
            "stt_time": timings.get("stt_time", 0),
            "translation_time": timings.get("translation_time", 0),
            "tts_time": timings.get("tts_time", 0),
            "merge_time": timings.get("merge_time", 0),
            "overall_time": overall_time
        }
        result_data["merged_media_url"] = f"http://{BASE_HOST}:8000/videos/{merged_filename}"

        return JSONResponse(content=result_data, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")


################################################################################
# ▶ /merge-media 엔드포인트 (별도 호출 시)
################################################################################
@app.post("/merge-media")
async def merge_media(
    video: List[UploadFile] = File(...),
    start_times: str = Form(...),
    red_track_indices: str = Form(...),
    audio: UploadFile = File(...)
):
    try:
        try:
            start_times_list = json.loads(start_times)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"start_times 파싱 실패: {str(e)}")

        try:
            red_indices = json.loads(red_track_indices)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"red_track_indices 파싱 실패: {str(e)}")

        if len(start_times_list) != len(video) or len(red_indices) != len(video):
            raise HTTPException(
                status_code=400,
                detail="비디오 파일 수와 start_times, red_track_indices 배열의 길이가 일치해야 합니다."
            )

        temp_folder = "temp"
        os.makedirs(temp_folder, exist_ok=True)

        video_clips_with_index = []
        for idx, vid in enumerate(video):
            video_temp_path = os.path.join(temp_folder, f"video_{int(time.time())}_{vid.filename}")
            with open(video_temp_path, "wb") as vf:
                vf.write(await vid.read())
            clip = VideoFileClip(video_temp_path)
            clip = clip.set_position((0, 0)).set_start(start_times_list[idx])
            if clip.audio:
                clip.audio = clip.audio.set_start(start_times_list[idx])
            video_clips_with_index.append({
                "clip": clip,
                "redIndex": red_indices[idx]
            })

        if not video_clips_with_index:
            raise HTTPException(status_code=400, detail="비디오 파일이 없습니다.")

        video_clips_with_index.sort(key=lambda x: x["redIndex"])
        video_clips_with_index.reverse()
        sorted_clips = [item["clip"] for item in video_clips_with_index]

        audio_temp_path = os.path.join(temp_folder, f"audio_{int(time.time())}_{audio.filename}")
        with open(audio_temp_path, "wb") as af:
            af.write(await audio.read())
        external_audio = AudioFileClip(audio_temp_path)

        composite_video = CompositeVideoClip(sorted_clips, size=sorted_clips[0].size)
        video_audio_clips = [clip.audio for clip in sorted_clips if clip.audio is not None]
        audio_list = video_audio_clips[:]
        audio_list.append(external_audio)
        if len(audio_list) > 0:
            composite_audio = CompositeAudioClip(audio_list)
            composite_video.audio = composite_audio

        output_path = os.path.join(temp_folder, f"merged_{int(time.time())}.mp4")
        composite_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        composite_video.close()
        for item in video_clips_with_index:
            item["clip"].close()
        external_audio.close()

        return FileResponse(path=output_path, filename="merged_output.mp4", media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merging failed: {str(e)}")
    
    
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
        video_url = f"{BASE_HOST}:8000/videos/{video[1]}"
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
 