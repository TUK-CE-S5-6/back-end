import os
import time
import json
import openai
import psycopg2
import requests
import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
    CompositeAudioClip
)

BASE_HOST = "host-url"  # 또는 배포 시 EC2 인스턴스의 공인 IP나 도메인

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

# Pydantic 모델 (사용자 관련 예시)
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    user_id: int
    username: str

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
############################################
async def transcribe_audio(audio_path: str, video_id: int):
    step_start = time.time()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"STT 변환 실패: {audio_path} 파일이 존재하지 않습니다.")
    
    result = clova_speech_stt(
        audio_path,
        completion="sync",
        language="ko-KR",
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
                (video_id, "ko", text, start_sec, end_sec, speaker)
            )
    else:
        curs.execute(
            """
            INSERT INTO transcripts (video_id, language, text, start_time, end_time, speaker)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (video_id, "ko", result.get("text", ""), 0, 0, "")
        )
    conn.commit()
    curs.close()
    conn.close()

    stt_time = time.time() - step_start
    return {"stt_time": stt_time}

############################################
# 번역 및 DB 저장
############################################
async def translate_video(video_id: int):
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
############################################
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    overall_start = time.time()  # 전체 처리 시작 시간 기록
    timings = {}

    try:
        original_file_name = file.filename
        file_name = os.path.splitext(original_file_name)[0]
        base_name = file_name[:-len("_audio")] if file_name.endswith("_audio") else file_name

        # 영상 파일 저장
        step_start = time.time()
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        timings["upload_time"] = time.time() - step_start

        # 영상 길이 계산 및 오디오 추출
        step_start = time.time()
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        extracted_audio_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(extracted_audio_path, codec='mp3')
        audio_clip.close()
        video_clip.close()
        timings["audio_extraction_time"] = time.time() - step_start

        # Spleeter 호출
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

        # DB에 비디오 정보 저장
        step_start = time.time()
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
        timings["db_time"] = time.time() - step_start

        # STT 및 번역 처리 (Clova Speech Long Sentence STT with diarization 옵션 사용)
        stt_timings = await transcribe_audio(separation_data.get("vocals_path"), video_id)
        timings.update(stt_timings)
        translation_timings = await translate_video(video_id)
        timings.update(translation_timings)

        # TTS 생성
        step_start = time.time()
        stt_data = {"video_id": video_id}
        tts_response = requests.post(f"{API_HOST}/generate-tts-from-stt", json=stt_data)
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS 생성 서비스 호출 실패")
        timings["tts_time"] = time.time() - step_start

        # 최종 결과 조회
        step_start = time.time()
        result_response = await get_edit_data(video_id)
        get_time = time.time() - step_start

        overall_time = time.time() - overall_start

        result_data = result_response.body if hasattr(result_response, "body") else {}
        if isinstance(result_data, bytes):
            result_data = json.loads(result_response.body.decode())
        result_data["timings"] = {
            "upload_time": timings.get("upload_time", 0),
            "audio_extraction_time": timings.get("audio_extraction_time", 0),
            "spleeter_time": timings.get("spleeter_time", 0),
            "db_time": timings.get("db_time", 0),
            "stt_time": timings.get("stt_time", 0),
            "translation_time": timings.get("translation_time", 0),
            "tts_time": timings.get("tts_time", 0),
            "get_time": get_time,
            "overall_time": overall_time
        }

        return JSONResponse(content=result_data, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

################################################################################
# ▶ /merge-media 엔드포인트
#    클라이언트가 WAV 파일과 여러 개의 비디오 파일 및 각 파일의 시작 시간 정보를 보내면,
#    서버에서 CompositeVideoClip/CompositeAudioClip을 이용해 최종 영상을 생성하여 반환합니다.
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
