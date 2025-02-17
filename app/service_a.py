import os
import time
import openai
import psycopg2
import requests
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from pyannote.audio import Pipeline
import json

# PostgreSQL 설정 (환경에 맞게 수정)
DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

# Clova Speech Long Sentence API 설정
# 아래 값을 네이버 클라우드 플랫폼에서 발급받은 Secret Key와 Invoke URL로 교체하세요.
NAVER_CLOVA_SECRET_KEY = "clova-key"  
NAVER_CLOVA_SPEECH_URL = "clova-url"

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

# Clova Speech Long Sentence STT 호출 함수 (Secret Key만 사용)
def clova_speech_stt(file_path: str, completion="sync", language="ko-KR", 
                     wordAlignment=True, fullText=True, diarization=None) -> str:
    """
    주어진 오디오 파일(file_path)을 Clova Speech Long Sentence API를 통해 전송하여,
    인식된 텍스트를 반환합니다.
    - completion: "sync" (동기) 또는 "async" (비동기)
    - language: 인식 언어 (예: "ko-KR")
    - wordAlignment, fullText: 옵션 (기본값 true)
    - diarization: 화자 인식 옵션 (없으면 기본값 사용)
    """
    # diarization 옵션이 제공되지 않은 경우 기본값 설정
    if diarization is None:
        diarization = {"enable": True, "speakerCountMin": -1, "speakerCountMax": -1}
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
    # 로컬 파일 업로드 방식: 파일과 함께 JSON 인코딩된 파라미터 전송
    files = {
        "media": open(file_path, "rb"),
        "params": (None, json.dumps(request_body).encode("UTF-8"), "application/json")
    }
    url = NAVER_CLOVA_SPEECH_URL + "/recognizer/upload"
    response = requests.post(url, headers=headers, files=files)
    files["media"].close()
    if response.status_code != 200:
        print("Clova Speech Long Sentence API 호출 실패:", response.status_code, response.text)
        return ""
    result = response.json()
    return result.get("text", "")

# -----------------------------------------------------------------
# 화자 다이어리제이션 함수 (pyannote.audio 사용)
# -----------------------------------------------------------------
def diarize_audio_eend(input_path: str):
    """
    입력 오디오 파일(input_path)을 받아 화자 다이어리제이션을 수행한 후,
    각 세그먼트의 시작/종료 시간과 화자 라벨을 포함한 리스트를 반환합니다.
    """
    # mp3이면 wav로 변환
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
# 각 세그먼트를 처리하는 함수 (Clova Speech Long Sentence STT 사용)
# -----------------------------------------------------------------
def process_segment(seg, audio, video_id):
    """
    각 세그먼트를 처리하여 Clova Speech Long Sentence STT 결과와 관련 정보를 반환합니다.
    """
    start_sec = seg["start"]
    end_sec = seg["end"]
    speaker = seg["speaker"]

    if end_sec - start_sec <= 0:
        return {"start": start_sec, "end": end_sec, "speaker": speaker, "text": ""}

    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)

    segment_audio = audio[start_ms:end_ms]
    temp_segment_path = os.path.join("temp_segments", f"{video_id}_{start_ms}_{end_ms}.wav")
    segment_audio.export(temp_segment_path, format="wav")

    # Clova Speech Long Sentence STT 호출 (동기 모드)
    text = clova_speech_stt(temp_segment_path, completion="sync").strip()

    os.remove(temp_segment_path)

    return {
        "start": start_sec,
        "end": end_sec,
        "speaker": speaker,
        "text": text,
    }

# -----------------------------------------------------------------
# STT 변환 (화자 다이어리제이션 + Clova Speech Long Sentence STT, 병렬 처리 적용)
# -----------------------------------------------------------------
async def transcribe_audio(audio_path: str, video_id: int):
    step_times = {}
    step_start = time.time()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"STT 변환 실패: {audio_path} 파일이 존재하지 않습니다.")
    
    diarization_start = time.time()
    segments = diarize_audio_eend(audio_path)
    step_times["diarization_time"] = time.time() - diarization_start

    audio = AudioSegment.from_file(audio_path)
    os.makedirs("temp_segments", exist_ok=True)

    tasks = [
        asyncio.to_thread(process_segment, seg, audio, video_id)
        for seg in segments
    ]
    segment_results = await asyncio.gather(*tasks)

    conn = get_connection()
    curs = conn.cursor()
    for res in segment_results:
        curs.execute(
            """
            INSERT INTO transcripts (video_id, language, text, start_time, end_time, speaker)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (video_id, "ko", res["text"], res["start"], res["end"], res["speaker"])
        )
    conn.commit()
    curs.close()
    conn.close()

    stt_time = time.time() - step_start
    step_times["stt_time"] = stt_time
    return step_times

# -----------------------------------------------------------------
# 번역 및 DB 저장
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# 최종 결과 조회 엔드포인트
# -----------------------------------------------------------------
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
    total_get_time = time.time() - step_start
    return JSONResponse(content={"video": video_data, "background_music": background_music, "tts_tracks": tts_tracks, "get_time": total_get_time}, status_code=200)

# -----------------------------------------------------------------
# 영상 업로드 및 처리 엔드포인트
# -----------------------------------------------------------------
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
                "http://localhost:8001/separate-audio",
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

        # STT 및 번역 처리 (화자 다이어리제이션 + Clova Speech Long Sentence STT)
        stt_timings = await transcribe_audio(separation_data.get("vocals_path"), video_id)
        timings.update(stt_timings)
        translation_timings = await translate_video(video_id)
        timings.update(translation_timings)

        # TTS 생성
        step_start = time.time()
        stt_data = {"video_id": video_id}
        tts_response = requests.post("http://localhost:8001/generate-tts-from-stt", json=stt_data)
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
