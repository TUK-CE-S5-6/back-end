import os
import time
import psycopg2
import logging
import requests
import librosa
import shutil
import subprocess
import json
from contextlib import contextmanager
from typing import Optional, List

import soundfile as sf  # 타임 스트레칭 후 오디오 저장용
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ----------------------------
# 기본 설정 및 환경 변수
# ----------------------------
logging.basicConfig(level=logging.DEBUG)

AUDIO_FOLDER = "extracted_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)
CUSTOM_TTS_FOLDER = os.path.join(AUDIO_FOLDER, "custom_tts")
os.makedirs(CUSTOM_TTS_FOLDER, exist_ok=True)
VOICE_MODEL_FOLDER = "voice_models"
os.makedirs(VOICE_MODEL_FOLDER, exist_ok=True)

# FastAPI 앱 생성
app = FastAPI()

DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

ELEVENLABS_API_KEY = "eleven-key"
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# CORS 설정
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Pydantic 모델
# ----------------------------
class CustomTTSRequest(BaseModel):
    tts_id: Optional[int] = None
    voice_id: str
    text: str

# ----------------------------
# PostgreSQL 연결 함수
# ----------------------------
def get_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

@contextmanager
def get_db_cursor():
    conn = get_connection()
    try:
        curs = conn.cursor()
        yield curs
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        curs.close()
        conn.close()

# ----------------------------
# 유틸리티 함수들
# ----------------------------
def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

def delete_temp_files(file_paths: List[str]):
    for fp in file_paths:
        try:
            if os.path.exists(fp):
                if os.path.isdir(fp):
                    shutil.rmtree(fp, ignore_errors=True)
                    logging.info(f"Deleted folder: {fp}")
                else:
                    os.remove(fp)
                    logging.info(f"Deleted file: {fp}")
        except Exception as e:
            logging.error(f"Failed to delete {fp}: {e}")

def run_spleeter_subprocess(input_path: str, output_folder: str):
    command = [
        "python", "-m", "spleeter.separator",
        "separate_to_file", input_path, output_folder,
        "-p", "spleeter:2stems"
    ]
    logging.info("Spleeter subprocess command: " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Spleeter subprocess failed: " + result.stderr)
    else:
        logging.info(result.stdout)

def find_spleeter_vocals(base_folder: str, base_name: str):
    vocals_candidates = []
    bgm_candidates = []
    primary_folder = os.path.join(base_folder, base_name)
    if os.path.exists(primary_folder):
        for root, dirs, files in os.walk(primary_folder):
            if "vocals.wav" in files:
                vocals_candidates.append(os.path.join(root, "vocals.wav"))
            if "accompaniment.wav" in files:
                bgm_candidates.append(os.path.join(root, "accompaniment.wav"))
    alternate_folder = os.path.join(base_folder, f"{base_name}_audio")
    if os.path.exists(alternate_folder):
        for root, dirs, files in os.walk(alternate_folder):
            if "vocals.wav" in files:
                vocals_candidates.append(os.path.join(root, "vocals.wav"))
            if "accompaniment.wav" in files:
                bgm_candidates.append(os.path.join(root, "accompaniment.wav"))
    if not vocals_candidates or not bgm_candidates:
        raise FileNotFoundError(
            f"❌ '{base_folder}' 내에 '{base_name}' 또는 '{base_name}_audio' 폴더에서 필요한 파일을 찾을 수 없습니다!"
        )
    # 여러 개 있더라도 첫 번째 항목만 선택해서 반환
    return vocals_candidates[0], bgm_candidates[0]


def split_audio(input_path: str, output_dir: str, max_size_mb: int = 10):
    ensure_folder(output_dir)
    try:
        audio = AudioSegment.from_file(input_path)
        threshold = audio.dBFS - 16
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=threshold, seek_step=1)
        parts = []
        for idx, (start, end) in enumerate(nonsilent_ranges):
            if end - start < 1000:
                continue
            segment = audio[start:end]
            part_path = os.path.join(output_dir, f"part_{idx}.mp3")
            segment.export(part_path, format="mp3", bitrate="192k")
            parts.append(part_path)
        logging.info(f"🔹 분할된 파일 개수: {len(parts)}")
        max_samples = 25
        if len(parts) > max_samples:
            indices = [int(i * (len(parts) - 1) / (max_samples - 1)) for i in range(max_samples)]
            parts = [parts[i] for i in indices]
            logging.info(f"🔹 샘플 수가 많아 균일하게 {max_samples}개로 축소함")
        return parts
    except Exception as e:
        logging.error(f"❌ 오디오 분할 실패: {str(e)}")
        return []

def merge_nonsilent_audio_improved(
    input_path: str,
    output_dir: str,
    min_silence_len: int = 500,
    silence_thresh: float = None,
    output_filename: str = "merged_sample.mp3",
    fade_duration: int = 200
):
    ensure_folder(output_dir)
    try:
        audio = AudioSegment.from_file(input_path)
        if silence_thresh is None:
            silence_thresh = audio.dBFS - 16
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=1
        )
        merged_audio = AudioSegment.empty()
        for start, end in nonsilent_ranges:
            if end - start < 1000:
                continue
            segment = audio[start:end]
            segment = segment.fade_in(fade_duration).fade_out(fade_duration)
            merged_audio += segment
        output_path = os.path.join(output_dir, output_filename)
        merged_audio.export(output_path, format="mp3", bitrate="192k")
        logging.info(f"🔹 병합된 파일 생성: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"❌ 병합 실패: {str(e)}")
        return None

def split_merged_audio(
    merged_path: str,
    output_dir: str,
    max_duration_sec: int = 30,
    max_samples: int = 25
):
    ensure_folder(output_dir)
    try:
        audio = AudioSegment.from_file(merged_path)
        duration_ms = len(audio)
        max_duration_ms = max_duration_sec * 1000
        parts = []
        if duration_ms <= max_duration_ms:
            parts.append(merged_path)
        else:
            num_parts = (duration_ms + max_duration_ms - 1) // max_duration_ms
            for i in range(num_parts):
                start = i * max_duration_ms
                end = min((i + 1) * max_duration_ms, duration_ms)
                segment = audio[start:end]
                part_path = os.path.join(output_dir, f"merged_part_{i}.mp3")
                segment.export(part_path, format="mp3", bitrate="192k")
                parts.append(part_path)
        if len(parts) > max_samples:
            indices = [
                int(i * (len(parts) - 1) / (max_samples - 1))
                for i in range(max_samples)
            ]
            parts = [parts[i] for i in indices]
            logging.info(f"🔹 병합 샘플 수가 많아 균일하게 {max_samples}개로 축소함")
        return parts
    except Exception as e:
        logging.error(f"❌ 병합 후 분할 실패: {str(e)}")
        return []

def create_voice_model_api(name: str, description: str, sample_file_paths: List[str]):
    url = f"{ELEVENLABS_BASE_URL}/voices/add"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    data = {"name": name, "description": description}
    files = []
    for path in sample_file_paths:
        try:
            f = open(path, "rb")
        except Exception as e:
            logging.error(f"파일 열기 실패 ({path}): {str(e)}")
            continue
        files.append(("files", (os.path.basename(path), f, "audio/mpeg")))
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        return response.json()
    finally:
        for _, file_tuple in files:
            file_tuple[1].close()

def adjust_tts_duration(file_path: str, desired_duration: float) -> float:
    y, sr = librosa.load(file_path, sr=None)
    current_duration = librosa.get_duration(y=y, sr=sr)
    rate = current_duration / desired_duration
    rate = min(max(rate, 0.85), 1.15)
    logging.info(
        f"adjust_tts_duration - 현재 길이: {current_duration:.2f} sec, "
        f"원하는 길이: {desired_duration:.2f} sec, 적용 비율: {rate:.2f}"
    )
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    sf.write(file_path, y_stretched, sr)
    new_duration = librosa.get_duration(y=y_stretched, sr=sr)
    logging.info(f"adjust_tts_duration - 조정 후 길이: {new_duration:.2f} sec")
    return new_duration

# ----------------------------
# FastAPI 엔드포인트
# ----------------------------
app.mount("/videos", StaticFiles(directory="uploaded_videos"), name="videos")
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")

@app.get("/")
def read_root():
    return {"message": "Hello from Service B (TTS Creation)!"}

@app.post("/separate-audio")
async def separate_audio(file: UploadFile = File(...)):
    try:
        original_name = os.path.splitext(file.filename)[0]
        base_name = (
            original_name[:-len("_audio")]
            if file.filename.endswith("_audio")
            else original_name
        )
        input_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        with open(input_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"입력 오디오 저장 완료: {input_path}")

        from spleeter.separator import Separator
        separator = Separator("spleeter:2stems")
        separator.separate_to_file(input_path, AUDIO_FOLDER)
        logging.info("Spleeter 분리 실행 완료")

        vocals_path, bgm_path = find_spleeter_vocals(AUDIO_FOLDER, base_name)
        logging.info(f"분리된 파일 찾음: vocals={vocals_path}, bgm={bgm_path}")

        return JSONResponse(
            content={
                "message": "Spleeter 분리 완료",
                "vocals_path": vocals_path,
                "bgm_path": bgm_path
            },
            status_code=200
        )
    except Exception as e:
        logging.error(f"Spleeter 처리 실패: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Spleeter 처리 실패: {str(e)}"
        )

@app.post("/generate-tts-from-stt")
async def generate_tts_from_stt(data: dict):
    try:
        video_id = data.get("video_id")
        if video_id is None:
            raise HTTPException(status_code=400, detail="video_id가 필요합니다.")
        with get_db_cursor() as curs:
            curs.execute(
                """
                SELECT t.translation_id, t.text, tr.start_time, tr.end_time
                FROM translations t
                JOIN transcripts tr ON t.transcript_id = tr.transcript_id
                WHERE tr.video_id = %s;
                """,
                (video_id,)
            )
            translations = curs.fetchall()
            if not translations:
                raise HTTPException(
                    status_code=404, detail="번역된 데이터가 없습니다."
                )
            tts_output_dir = os.path.join(AUDIO_FOLDER, f"{video_id}_tts")
            ensure_folder(tts_output_dir)
            selected_voice_id = "5Af3x6nAIWjF6agOOtOz"
            for translation_id, text, start_time, end_time in translations:
                try:
                    url = (
                        f"{ELEVENLABS_BASE_URL}/text-to-speech/"
                        f"{selected_voice_id}?output_format=mp3_44100_128"
                    )
                    headers = {
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "text": text,
                        "model_id": "eleven_multilingual_v2",
                    }
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise Exception(
                            f"ElevenLabs TTS 생성 실패: {response.text}"
                        )
                    audio_content = response.content

                    tts_audio_path = os.path.join(
                        tts_output_dir, f"{translation_id}.mp3"
                    )
                    with open(tts_audio_path, "wb") as tts_file:
                        tts_file.write(audio_content)

                    current_duration = librosa.get_duration(path=tts_audio_path)
                    desired_duration = float(end_time) - float(start_time)

                    # 로그 추가: TTS 텍스트, 원본 길이, 원하는 길이 출력
                    logging.info(f"TTS 텍스트: {text}")
                    logging.info(
                        f"원본 TTS 길이: {current_duration:.2f} sec, "
                        f"원하는 길이: {desired_duration:.2f} sec"
                    )

                    new_duration = adjust_tts_duration(
                        tts_audio_path, desired_duration
                    )

                    # 로그 추가: 조정 후 길이 출력
                    logging.info(
                        f"조정 후 TTS 길이: {new_duration:.2f} sec "
                        f"(원본: {current_duration:.2f} sec "
                        f"→ 조정: {new_duration:.2f} sec)"
                    )

                    curs.execute(
                        """
                        INSERT INTO tts
                        (translation_id, file_path, voice, start_time, duration)
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        (
                            translation_id,
                            tts_audio_path,
                            selected_voice_id,
                            float(start_time),
                            float(new_duration),
                        ),
                    )
                except Exception as e:
                    logging.error(f"TTS 생성 실패: {str(e)}")
        return JSONResponse(
            content={"message": "TTS 생성 완료"}, status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 처리 실패: {str(e)}")


@app.post("/generate-tts")
async def generate_tts_custom(request: CustomTTSRequest):
    try:
        conn = get_connection()
        curs = conn.cursor()
        with get_db_cursor() as curs:
            if request.tts_id is not None:
                curs.execute(
                    """
                    SELECT translation_id, file_path
                    FROM tts
                    WHERE tts_id = %s;
                    """,
                    (request.tts_id,),
                )
                row = curs.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail="해당 tts_id를 찾을 수 없습니다.",
                    )
                translation_id, tts_audio_path = row
                curs.execute(
                    """
                    UPDATE translations
                    SET text = %s
                    WHERE translation_id = %s;
                    """,
                    (request.text, translation_id),
                )
                try:
                    url = (
                        f"{ELEVENLABS_BASE_URL}/text-to-speech/"
                        f"{request.voice_id}?output_format=mp3_44100_128"
                    )
                    headers = {
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "text": request.text,
                        "model_id": "eleven_multilingual_v2",
                    }
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise Exception(
                            f"ElevenLabs TTS 생성 실패: {response.text}"
                        )
                    audio_content = response.content
                    with open(tts_audio_path, "wb") as tts_file:
                        tts_file.write(audio_content)
                    duration = librosa.get_duration(path=tts_audio_path)
                    curs.execute(
                        """
                        UPDATE tts
                        SET voice = %s, duration = %s
                        WHERE tts_id = %s;
                        """,
                        (request.voice_id, duration, request.tts_id),
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"ElevenLabs TTS 생성 실패: {str(e)}",
                    )
            else:
                timestamp = int(time.time())
                tts_audio_path = os.path.join(
                    CUSTOM_TTS_FOLDER, f"tts_{timestamp}.mp3"
                )
                try:
                    url = (
                        f"{ELEVENLABS_BASE_URL}/text-to-speech/"
                        f"{request.voice_id}?output_format=mp3_44100_128"
                    )
                    headers = {
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "text": request.text,
                        "model_id": "eleven_multilingual_v2",
                    }
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise Exception(
                            f"ElevenLabs TTS 생성 실패: {response.text}"
                        )
                    audio_content = response.content
                    with open(tts_audio_path, "wb") as tts_file:
                        tts_file.write(audio_content)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"ElevenLabs TTS 생성 실패: {str(e)}",
                    )
                curs.execute(
                    """
                    INSERT INTO translations (text, language)
                    VALUES (%s, %s)
                    RETURNING translation_id;
                    """,
                    (request.text, "en"),
                )
                translation_id = curs.fetchone()[0]
                duration = librosa.get_duration(path=tts_audio_path)
                curs.execute(
                    """
                    INSERT INTO tts
                    (translation_id, file_path, voice, start_time, duration)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING tts_id;
                    """,
                    (
                        translation_id,
                        tts_audio_path,
                        request.voice_id,
                        0.0,
                        duration,
                    ),
                )
                request.tts_id = curs.fetchone()[0]
        conn.commit()
        curs.close()
        conn.close()
        tts_file_url = tts_audio_path.replace(AUDIO_FOLDER, "/extracted_audio")
        return JSONResponse(
            content={
                "message": "TTS 생성 완료",
                "file_url": tts_file_url,
                "tts_id": request.tts_id,
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 처리 실패: {str(e)}")


@app.post("/create-voice-model")
async def create_voice_model(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    업로드된 여러 오디오 파일을 받아서,
    각 파일에 대해 Spleeter로 보컬(음성)만 분리한 후,
    분리된 보컬 음원을 기반으로 무음 구간을 제거하고 병합/분할하여
    최대 25개의 샘플 파일을 생성합니다.
    이 샘플 파일들을 ElevenLabs API에 전달하여 보이스 모델(클론)을 생성하고,
    생성된 모델 정보를 DB에 저장합니다.
    작업 완료 후 임시 파일들은 BackgroundTasks를 이용하여 삭제됩니다.
    
    **파일 크기는 10MB 이하만 허용됩니다.**
    """
    all_sample_parts = []
    temp_paths = []

    MAX_FILE_SIZE = 10 * 1024 * 1024

    for file in files:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"파일 {file.filename}의 크기가 10MB를 초과합니다."
            )

        original_name = os.path.splitext(file.filename)[0]
        base_name = (
            original_name[:-len("_audio")]
            if file.filename.endswith("_audio")
            else original_name
        )

        # 파일명을 그대로 사용 (시간 제거)
        original_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        logging.info(f"📥 파일 저장 시작: {file.filename}")
        with open(original_path, "wb") as f:
            f.write(await file.read())
        temp_paths.append(original_path)

        # Spleeter 실행
        try:
            from spleeter.separator import Separator
            separator = Separator("spleeter:2stems")
            separator.separate_to_file(original_path, AUDIO_FOLDER)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Spleeter 실행 실패: {str(e)}"
            )

        # vocals.wav 파일 찾기
        vocal_path, _ = find_spleeter_vocals(AUDIO_FOLDER, base_name)

        # 고정된 이름으로 보컬 파일 저장
        fixed_vocal_path = os.path.join(AUDIO_FOLDER, f"{base_name}_vocals.wav")
        shutil.move(vocal_path, fixed_vocal_path)
        temp_paths.append(fixed_vocal_path)

        # Spleeter가 생성한 원본 폴더 삭제
        folder_primary = os.path.join(AUDIO_FOLDER, base_name)
        if os.path.exists(folder_primary):
            temp_paths.append(folder_primary)
            shutil.rmtree(folder_primary, ignore_errors=True)

        folder_alternate = os.path.join(AUDIO_FOLDER, f"{base_name}_audio")
        if os.path.exists(folder_alternate):
            temp_paths.append(folder_alternate)
            shutil.rmtree(folder_alternate, ignore_errors=True)

        # 병합 및 분할
        merge_dir = os.path.join(AUDIO_FOLDER, f"{base_name}_merged")
        os.makedirs(merge_dir, exist_ok=True)
        temp_paths.append(merge_dir)
        merged_sample_path = merge_nonsilent_audio_improved(
            fixed_vocal_path, merge_dir,
            output_filename="merged_sample.mp3",
            fade_duration=200
        )
        if not merged_sample_path:
            raise HTTPException(
                status_code=500, detail="병합 파일 생성 실패"
            )

        split_dir = os.path.join(AUDIO_FOLDER, f"{base_name}_split")
        os.makedirs(split_dir, exist_ok=True)
        temp_paths.append(split_dir)
        MAX_MERGED_DURATION_SEC = 30
        sample_parts = split_merged_audio(
            merged_sample_path, split_dir,
            max_duration_sec=MAX_MERGED_DURATION_SEC,
            max_samples=25
        )
        if not sample_parts:
            raise HTTPException(
                status_code=500, detail="병합 후 분할 실패"
            )

        all_sample_parts.extend(sample_parts)

    if not all_sample_parts:
        raise HTTPException(
            status_code=500, detail="샘플 파일 생성 실패"
        )

    # ElevenLabs API 호출
    voice_response = create_voice_model_api(
        name=name,
        description=description,
        sample_file_paths=all_sample_parts
    )
    voice_id = voice_response.get("voice_id")
    if not voice_id:
        raise Exception("보이스 모델 생성 실패: voice_id가 반환되지 않음")
    logging.info(f"✅ 보이스 모델 생성 완료: {voice_id}")

    with get_db_cursor() as curs:
        curs.execute(
            """
            INSERT INTO voice_models (voice_id, name, description)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (voice_id, name, description),
        )
        inserted_id = curs.fetchone()[0]
    logging.info(f"📌 DB에 voice_models 테이블에 저장 완료 (id: {inserted_id})")

    # 임시 파일 삭제
    background_tasks.add_task(delete_temp_files, temp_paths)

    return JSONResponse(
        content={
            "message": "보이스 모델 생성 완료",
            "voice_id": voice_id,
            "name": name,
            "description": description,
            "db_id": inserted_id,
        },
        status_code=200
    )
