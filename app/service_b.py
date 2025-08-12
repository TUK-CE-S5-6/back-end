import os
import time
import psycopg2
import logging
import requests
import librosa
import shutil
import subprocess
import json
import openai
from contextlib import contextmanager
from typing import Optional, List
from openai.error import RateLimitError, APIError, ServiceUnavailableError, Timeout
from openai import Audio  # Whisper 호출용

import soundfile as sf  # 타임 스트레칭 후 오디오 저장용
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from spleeter.separator import Separator

import regex as re     # pip install regex   ← 유니코드 그랩이 더 편해요
import unicodedata
from uuid import uuid4
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageDraw  # 이미 import 되어있음
import librosa, numpy as np

# ── NEW ───────────────────────────────────────────────
_HIRA_RE  = re.compile(r'[\p{Hiragana}]')
_KATA_RE  = re.compile(r'[\p{Katakana}]')
_CHOON_RE = re.compile(r'ー')           # 장음부호
_SMALL_RE = re.compile(r'[ゃゅょァィゥェォャュョヮぁぃぅぇぉゎ]')  # 축약음
_EN_RE    = re.compile(r'[A-Za-z0-9]')  # 영어·숫자(0.5 자 가중)

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
USER_FILES_FOLDER = "user_files"
os.makedirs(USER_FILES_FOLDER, exist_ok=True)
IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
WAVEFORM_FOLDER = "waveforms"
os.makedirs(WAVEFORM_FOLDER, exist_ok=True)

# FastAPI 앱 생성
app = FastAPI()

DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

ELEVENLABS_API_KEY = "eleven-key"
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# OpenAI API 설정 (번역용)
OPENAI_API_KEY = "gpt-key"
openai.api_key = OPENAI_API_KEY


# ── NEW: 언어별 평균 읽기 속도 (음절/문자 per sec)
AVG_SYL_PER_SEC = {"ko": 6.2, "en": 6.19, "ja": 7.84, "zh": 5.18}
TOL_RATIO = 0.02
MIN_SENT_LEN = 4
SUPPORTED_LANGS = set(AVG_SYL_PER_SEC)  # {'ko','en','ja','zh'}
RETRY_STATUS = {429, 500, 502, 503, 504}
# ── 언어별 TTS 길이 허용 오차 (실제 길이 ↔ 목표 길이 비율) ──
LANG_TOL = {
    "ko": 0.1,   # 음절·호흡 고려
    "en": 0.05,
    "ja": 0.1,   # 모라 단위 특성상 5% 이내 맞추기 난이도 ↑
    "zh": 0.08,
}
def get_tol(lang_code: str) -> float:
    """언어 코드 앞 두 글자로 TOL 반환, 기본 5%"""
    return LANG_TOL.get(lang_code[:2], 0.05)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


def gpt_call_with_retry(
    model,
    messages,
    max_retries=6,
    initial_delay=0.8,
    request_timeout=20,
    temperature=0.2,
    fallback_models=("gpt-4o-mini", "gpt-4-turbo"),
    **kwargs
):
    """
    OpenAI ChatCompletion 지수 백오프 + 지터 재시도.
    마지막엔 폴백 모델로도 시도.
    """
    import random, time
    delay = initial_delay
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            return openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=request_timeout,
                **kwargs
            )
        except (RateLimitError, ServiceUnavailableError, Timeout, APIError) as e:
            last_err = e
            status = getattr(e, "http_status", None)
            # 재시도 대상이 아니면 즉시 중단
            if status not in RETRY_STATUS and not isinstance(e, (Timeout,)):
                logging.error(f"[GPT] non-retryable: {e}")
                break
            if attempt == max_retries:
                break
            sleep_s = delay * (1 + random.random() * 0.5)  # 지터
            logging.warning(f"[GPT] retry {attempt}/{max_retries} in {sleep_s:.2f}s (err={e})")
            time.sleep(sleep_s)
            delay *= 2
        except Exception as e:
            last_err = e
            logging.exception(f"[GPT] unexpected: {e}")
            break

    # 폴백 모델들
    for fb in fallback_models:
        try:
            logging.warning(f"[GPT] fallback model: {fb}")
            return openai.ChatCompletion.create(
                model=fb,
                messages=messages,
                temperature=temperature,
                request_timeout=request_timeout,
                **kwargs
            )
        except Exception as e:
            last_err = e
            logging.warning(f"[GPT] fallback {fb} failed: {e}")

    raise last_err or RuntimeError("OpenAI call failed")

def _mora_count(text: str) -> int:
    """
    히라가나·가타카나·한자 기준으로 모라(mora)를 근사 계산.
    - 장음(ー)·축약음(ゃ 등)은 0.5 로 가중
    - 영숫자는 0.5 로 가중 (예: "ABC" ≈ 1.5 mora)
    """
    cnt  = len(_HIRA_RE.findall(text)) + len(_KATA_RE.findall(text))
    cnt += len(re.findall(r'[\p{Han}]', text))          # 한자(=1)
    cnt += 0.5 * (len(_CHOON_RE.findall(text)) +
                  len(_SMALL_RE.findall(text)) +
                  len(_EN_RE.findall(text)))
    return int(round(cnt))

def _weighted_len_zh(text: str) -> float:
    """중국어: 한자=1, 영숫자=0.5, 그 외=0.3 로 근사"""
    han   = len(re.findall(r'[\p{Han}]', text))
    ennum = len(_EN_RE.findall(text))
    other = len(text) - han - ennum
    return han + 0.5*ennum + 0.3*other
# ──────────────────────────────────────────────────────

def count_units(txt: str, lang: str) -> int:
    """
    TTS 길이 예측용 단위 계산
    ko : “완성형 한글 음절”만 카운트
    ja : 모라(mora) 근사
    zh : 가중치 한자수
    기타(en 등) : 문자수
    """
    if lang == "ko":
        # 1) NFD 정규화: 완성형→초중종성 자모로 분해
        norm = unicodedata.normalize("NFD", txt)
        # 2) 완성형 음절 개수
        syllables = sum(1 for c in norm if "\uAC00" <= c <= "\uD7A3")
        # 3) 자모 개수 (Hangul Jamo & Compatibility Jamo)
        jamo = sum(1 for c in norm if (
            "\u1100" <= c <= "\u11FF" or   # Hangul Jamo
            "\u3130" <= c <= "\u318F"      # Compatibility Jamo
        ))
        cnt = syllables + 0.5 * jamo
        return int(round(cnt))

    if lang == "ja":
        return _mora_count(txt)

    if lang == "zh":
        return int(round(_weighted_len_zh(txt)))

    return len(txt)

def translate_length_aware(
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    dur_sec: float,
    voice_id: str,
    db_cursor,
    retries: int = 6,
    tol_ratio: float = 0.03,   # 목표 유닛 대비 ±3%
    step_ratio: float = 0.05   # 목표 유닛의 5%만큼 goal 보정
) -> str:
    """
    길이-의식 번역(텍스트 재작성 없이): 가장 근접한 번역문을 반환.
    """
    tgt = tgt_lang[:2]
    cps = get_cps(db_cursor, voice_id, tgt)       # voice_lang_stats 없으면 fallback 사용
    target = max(1, int(round(dur_sec * cps)))    # 목표 유닛 수
    tol    = max(2, int(round(target * tol_ratio)))
    lo, hi = target - tol, target + tol
    step   = max(1, int(round(target * step_ratio)))
    goal   = target

    best_text = None
    best_dev  = 10**9

    def _dev(txt: str) -> int:
        return abs(count_units(txt, tgt) - target)

    for _ in range(retries):
        sys_msg = (
            f"Translate from {src_lang[:2]} to {tgt}. "
            f"Make the output exactly {goal} units long. "
            "Keep proper nouns; do NOT summarize. "
            "Return ONLY the translation."
        )
        try:
            res = gpt_call_with_retry(
                model="gpt-4o",
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user",   "content": src_text}],
                request_timeout=15
            ).choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"[translate_length_aware] retryable error: {e}")
            continue

        if not res:
            continue

        d = _dev(res)
        if d < best_dev:
            best_dev, best_text = d, res

        units = count_units(res, tgt)
        if lo <= units <= hi:
            return res  # 허용오차 내면 즉시 종료

        # 목표(goal) 보정만 수행(텍스트는 확정하지 않음)
        goal += step if units < lo else -step
        goal = max(lo, min(hi, goal))

    # 모든 시도에서 허용오차 실패 → 가장 근접한 후보 사용(없으면 원문)
    return (best_text or src_text).strip()


# ― Stretch (±3 % 이내)
def stretch_audio(input_path: str, output_path: str, current_duration: float, desired_duration: float) -> float:
    speed = current_duration / desired_duration
    speed = max(0.90, min(1.03, speed))  # ±3 %
    logging.debug(f"Speed change {speed:.3f}×")

    sound = AudioSegment.from_file(input_path)
    new_fr = int(sound.frame_rate * speed)
    stretched = sound._spawn(sound.raw_data, overrides={"frame_rate": new_fr})
    stretched = stretched.set_frame_rate(sound.frame_rate)
    ensure_folder(os.path.dirname(output_path))
    stretched.export(output_path, format="mp3", bitrate="192k")
    return len(stretched) / 1000.0

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

def adjust_tts_duration(file_path: str, desired_duration: float, max_change: float = 0.25) -> float:
    """
    생성된 TTS 파일을 목표 길이에 맞게 시간 스트레칭(피치 보존).
    max_change=0.25 → 0.8~1.25배 범위로 제한.
    """
    y, sr = librosa.load(file_path, sr=None)
    current_duration = librosa.get_duration(y=y, sr=sr)
    if desired_duration <= 0 or current_duration <= 0:
        return current_duration

    rate = current_duration / desired_duration           # >1이면 더 빠르게(짧아짐)
    lo, hi = (1.0 / (1.0 + max_change)), (1.0 + max_change)
    rate = min(max(rate, lo), hi)

    logging.info(
        f"[adjust_tts_duration] cur={current_duration:.3f}s -> tgt={desired_duration:.3f}s, rate={rate:.3f}"
    )

    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    sf.write(file_path, y_stretched, sr)
    new_duration = librosa.get_duration(y=y_stretched, sr=sr)
    logging.info(f"[adjust_tts_duration] new={new_duration:.3f}s")
    return new_duration

# ----------------------------
# FastAPI 엔드포인트
# ----------------------------
app.mount("/uploaded_videos", StaticFiles(directory="uploaded_videos"), name="videos")
app.mount("/extracted_audio", StaticFiles(directory="extracted_audio"), name="audio")
app.mount("/images", StaticFiles(directory=IMAGE_FOLDER), name="images")
app.mount("/waveforms", StaticFiles(directory=WAVEFORM_FOLDER), name="waveforms")


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

def get_cps(cur, voice_id: str, lang: str, fallback: float = 6.1) -> float:
    """
    voice_lang_stats 테이블에서 (voice_id, lang) 행의 char_per_sec를 반환.
    없으면 fallback 값(기본 6.1) 사용.
    """
    cur.execute(
        "SELECT char_per_sec FROM voice_lang_stats WHERE voice_id=%s AND lang=%s;",
        (voice_id, lang[:2])
    )
    row = cur.fetchone()
    return row[0] if row else fallback

# ────────────────────────────────────────────────────────────
#  공통: ElevenLabs 호출 래퍼
# ────────────────────────────────────────────────────────────
# --- helper: 속도 인자 지원 버전 ---
def synth_to_file(
    voice_id: str,
    text: str,
    out_path: str,
    speed: float | None = None,
    model_id: str = "eleven_multilingual_v2",
):
    payload = {"text": text, "model_id": model_id}
    if speed is not None:
        sp = max(0.7, min(1.2, float(speed)))  # 권장 범위
        payload["voice_settings"] = {"speed": sp}

    resp = requests.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}?output_format=mp3_44100_128",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)


# ───────────────────────────────────────────────────────────
#  /generate-tts-from-stt
# ───────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────
#  /generate-tts-from-stt
# ───────────────────────────────────────────────────────────
@app.post("/generate-tts-from-stt")
async def generate_tts_from_stt(data: dict):
    video_id = data.get("video_id")
    if not video_id:
        raise HTTPException(400, "video_id 필요")

    speaker_voice = {
        "A": "29vD33N1CtxCmqQRPOHJ",
        "B": "21m00Tcm4TlvDq8ikWAM",
        "C": "5Q0t7uMcjvnagumLfvZi",
    }
    MAX_REWRITE_RETRY = 5
    tts_dir = os.path.join(AUDIO_FOLDER, f"{video_id}_tts")
    ensure_folder(tts_dir)

    failed: list[dict] = []
    generated_cnt = 0

    with get_db_cursor() as cur:
        cur.execute(
            """
            SELECT tr.transcript_id, tr.text, tr.start_time, tr.end_time,
                   tr.speaker,
                   LEFT(p.source_language,2) AS src_l,
                   LEFT(p.target_language,2) AS tgt_l
              FROM transcripts tr
              JOIN videos   v ON v.video_id = tr.video_id
              JOIN projects p ON p.project_id = v.project_id
             WHERE tr.video_id = %s
             ORDER BY tr.start_time ASC;
            """,
            (video_id,),
        )
        rows = cur.fetchall()

        for tid, src, st, et, spk, src_l, tgt_l in rows:
            try:
                dur   = float(et) - float(st)
                voice = speaker_voice.get(spk)
                if dur <= 0 or not voice:
                    continue

                TOL = get_tol(tgt_l)

                # ① 길이-의식 번역(시드)
                text = translate_length_aware(
                    src_text=src,
                    src_lang=src_l,
                    tgt_lang=tgt_l,
                    dur_sec=dur,
                    voice_id=voice,
                    db_cursor=cur,
                )

                # ② 합성/검증 루프 + "가장 근접" 후보 기억
                best_delta = float("inf")
                best_mp3   = None
                best_dur   = None
                best_text  = None

                chosen_path = None
                chosen_dur  = None
                chosen_text = None

                for n in range(MAX_REWRITE_RETRY):
                    tmp_path = os.path.join(tts_dir, f"tmp_{tid}_{n}.mp3")
                    synth_to_file(voice, text, tmp_path)

                    real = librosa.get_duration(path=tmp_path) or 0.0
                    delta = abs(real - dur)

                    if delta < best_delta:
                        best_delta, best_mp3, best_dur, best_text = delta, tmp_path, real, text

                    if dur > 0 and abs(real - dur) / dur <= TOL:
                        chosen_path, chosen_dur, chosen_text = tmp_path, real, text
                        break

                    # 다음 시도를 위한 텍스트 길이 재작성
                    safe_real = real if real > 0 else 1e-6
                    need_len  = max(8, round(len(text) * dur / safe_real))
                    unit_nm   = (
                        'syllables' if tgt_l == 'ko'
                        else 'mora' if tgt_l == 'ja'
                        else 'han characters' if tgt_l == 'zh'
                        else 'chars'
                    )
                    sys_msg = f"Rewrite to {need_len} {unit_nm}. Keep proper nouns; do NOT summarize."
                    try:
                        text = gpt_call_with_retry(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": sys_msg},
                                {"role": "user",   "content": text},
                            ],
                            request_timeout=15,
                        ).choices[0].message.content.strip()
                    except Exception as e:
                        logging.warning(f"[rewrite] GPT 실패(계속 진행): {e}")

                # ③ 8회 모두 실패 → '가장 근접' 텍스트로 speed/SSML/타임스트레치 순 폴백
                if chosen_path is None:
                    if not best_mp3:
                        raise RuntimeError("합성에 실패하여 후보가 없습니다.")

                    chosen_text = best_text
                    chosen_dur  = best_dur
                    chosen_path = best_mp3

                    try:
                        if best_dur and dur > 0:
                            needed = dur / best_dur  # 필요한 속도 배율
                            if 0.7 <= needed <= 1.2:
                                speed_path = os.path.join(tts_dir, f"best_{tid}_spd.mp3")
                                synth_to_file(voice, best_text, speed_path, speed=needed)
                                real_spd = librosa.get_duration(path=speed_path) or best_dur
                                chosen_path, chosen_dur = speed_path, real_spd
                            else:
                                rate_pct = int((dur / (best_dur or dur or 1.0)) * 100)
                                ssml = f'<speak><prosody rate="{rate_pct}%">{best_text}</prosody></speak>'
                                ssml_path = os.path.join(tts_dir, f"best_{tid}_ssml.mp3")
                                synth_to_file(voice, ssml, ssml_path)
                                chosen_path = ssml_path
                                chosen_dur  = librosa.get_duration(path=ssml_path) or best_dur
                    except Exception as e:
                        logging.warning(f"[speed/ssml-fallback] 실패: {e}")
                        try:
                            new_d = adjust_tts_duration(chosen_path, dur, max_change=0.25)
                            chosen_dur = new_d
                        except Exception as e2:
                            logging.warning(f"[adjust-fallback] 실패(그냥 진행): {e2}")

                final_mp3, final_dur, final_text = chosen_path, chosen_dur, chosen_text

                # ④ translations UPSERT (저장 텍스트 = chosen/best 텍스트)
                cur.execute(
                    """
                    INSERT INTO translations (transcript_id, language, text)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (transcript_id, language)
                    DO UPDATE SET text = EXCLUDED.text
                    RETURNING translation_id;
                    """,
                    (tid, tgt_l, final_text),
                )
                translation_id = cur.fetchone()[0]

                # ⑤ 파형 생성 + tts INSERT/UPDATE (STT 시간값 보존)
                wave_name = f"tts_{tid}.png"
                wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
                width_px  = int(max(1, round(float(final_dur) * 100)))
                generate_waveform_png(final_mp3, wave_path, width_px=width_px, height_px=100)
                wave_url = f"/waveforms/{wave_name}"

                cur.execute("SELECT tts_id FROM tts WHERE translation_id = %s;", (translation_id,))
                row = cur.fetchone()
                if row:
                    cur.execute(
                        """
                        UPDATE tts
                           SET file_path=%s, voice=%s, duration=%s, waveform_url=%s
                         WHERE tts_id=%s;
                        """,
                        (final_mp3, voice, final_dur, wave_url, row[0]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO tts (translation_id, file_path, voice, start_time, duration, waveform_url)
                        VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (translation_id, final_mp3, voice, st, final_dur, wave_url),
                    )

                generated_cnt += 1

            except Exception as e:
                logging.error(f"[generate_tts_from_stt] segment {tid} 실패: {e}")
                failed.append({"transcript_id": tid, "error": str(e)})
                continue

    if failed:
        return JSONResponse(
            status_code=207,
            content={"detail": "partial", "generated": generated_cnt, "failed": failed}
        )
    return {"message": "TTS 완료", "generated": generated_cnt}


# ───────────────────────────────────────────────────────────
#  /edit-tts
# ───────────────────────────────────────────────────────────
@app.post("/edit-tts")
async def edit_tts(
    tts_id: int = Form(...),
    voice: str = Form(...),
    text: str = Form(...),
    update_src: bool = Form(True),
):
    MAX_RETRY = 5

    with get_db_cursor() as cur:
        cur.execute(
            """
            SELECT  tts.translation_id,
                    ts.transcript_id,
                    tts.start_time,
                    tts.duration,
                    tts.file_path,
                    LEFT(p.source_language,2) AS src_l,
                    LEFT(p.target_language,2) AS tgt_l
              FROM tts
              JOIN translations tr ON tr.translation_id = tts.translation_id
              JOIN transcripts ts     ON ts.transcript_id     = tr.transcript_id
              JOIN videos v           ON v.video_id            = ts.video_id
              JOIN projects p         ON p.project_id          = v.project_id
             WHERE tts.tts_id = %s;
            """,
            (tts_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="TTS 레코드가 없습니다.")

        old_tran_id, transcript_id, start_time, target_dur, old_path, src_l, tgt_l = row
        base_dir = os.path.dirname(old_path)
        ensure_folder(base_dir)

        TOL = get_tol(tgt_l)

        # 길이-의식 번역(시드)
        new_text = translate_length_aware(
            src_text=text,
            src_lang=src_l,
            tgt_lang=tgt_l,
            dur_sec=target_dur,
            voice_id=voice,
            db_cursor=cur,
        )

        best_delta = float("inf")
        best_mp3   = None
        best_dur   = None
        best_text  = None

        final_path = None
        final_dur  = None
        final_text = None

        for n in range(MAX_RETRY):
            tmp_file = os.path.join(base_dir, f"edit_{tts_id}_{n}.mp3")
            synth_to_file(voice, new_text, tmp_file)

            real_dur = librosa.get_duration(path=tmp_file) or 0.0
            delta = abs(real_dur - target_dur)

            if delta < best_delta:
                best_delta, best_mp3, best_dur, best_text = delta, tmp_file, real_dur, new_text

            if target_dur > 0 and abs(real_dur - target_dur) / target_dur <= TOL:
                final_path, final_dur, final_text = tmp_file, real_dur, new_text
                break

            # 재작성
            need_len = max(8, round(len(new_text) * target_dur / max(real_dur, 1e-6)))
            unit_nm = (
                'syllables' if tgt_l == 'ko'
                else 'mora' if tgt_l == 'ja'
                else 'han characters' if tgt_l == 'zh'
                else 'chars'
            )
            sys_msg = f"Rewrite to {need_len} {unit_nm}. Keep names."
            try:
                new_text = gpt_call_with_retry(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user",   "content": new_text},
                    ],
                    request_timeout=15,
                ).choices[0].message.content.strip()
            except Exception as e:
                logging.warning(f"[edit-tts/rewrite] GPT 실패(계속): {e}")

        # 최대 시도 초과 → '가장 근접' 텍스트로 speed/SSML/타임스트레치 순 폴백
        if final_path is None:
            if not best_mp3:
                raise RuntimeError("합성에 실패하여 후보가 없습니다.")

            final_text = best_text
            final_dur  = best_dur
            final_path = best_mp3

            try:
                if best_dur and target_dur > 0:
                    needed = target_dur / best_dur
                    if 0.7 <= needed <= 1.2:
                        speed_file = os.path.join(base_dir, f"edit_{tts_id}_best_spd.mp3")
                        synth_to_file(voice, best_text, speed_file, speed=needed)
                        real_spd = librosa.get_duration(path=speed_file) or best_dur
                        final_path, final_dur = speed_file, real_spd
                    else:
                        rate_pct = int((target_dur / (best_dur or target_dur or 1.0)) * 100)
                        ssml = f'<speak><prosody rate="{rate_pct}%">{best_text}</prosody></speak>'
                        ssml_file = os.path.join(base_dir, f"edit_{tts_id}_best_ssml.mp3")
                        synth_to_file(voice, ssml, ssml_file)
                        final_path = ssml_file
                        final_dur  = librosa.get_duration(path=ssml_file) or best_dur
            except Exception as e:
                logging.warning(f"[edit/speed-ssml-fallback] 실패: {e}")
                try:
                    new_d = adjust_tts_duration(final_path, target_dur, max_change=0.25)
                    final_dur = new_d
                except Exception as e2:
                    logging.warning(f"[edit/adjust-fallback] 실패(그냥 진행): {e2}")

        # translations UPSERT (저장 텍스트 = 최종 선택 텍스트)
        cur.execute(
            """
            INSERT INTO translations (transcript_id, language, text)
            VALUES (%s, %s, %s)
            ON CONFLICT (transcript_id, language)
            DO UPDATE SET text = EXCLUDED.text
            RETURNING translation_id;
            """,
            (transcript_id, tgt_l, final_text),
        )
        new_translation_id = cur.fetchone()[0]

        # 파형 갱신
        wave_name = f"tts_{tts_id}.png"
        wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
        width_px  = int(max(1, round(float(final_dur) * 100)))
        generate_waveform_png(final_path, wave_path, width_px=width_px, height_px=100)
        wave_url = f"/waveforms/{wave_name}"

        # tts 업데이트 (STT 시간값 불변)
        cur.execute(
            """
            UPDATE tts
               SET translation_id = %s,
                   file_path      = %s,
                   voice          = %s,
                   duration       = %s,
                   waveform_url   = %s
             WHERE tts_id        = %s;
            """,
            (new_translation_id, final_path, voice, final_dur, wave_url, tts_id),
        )

        if update_src:
            cur.execute(
                "UPDATE transcripts SET text = %s WHERE transcript_id = %s;",
                (text, transcript_id),
            )

    return {
        "id": tts_id,
        "duration": final_dur,
        "url": final_path,
        "translation_id": new_translation_id,
        "translateText": final_text,
        "originalText": text,
        "waveform_url": wave_url,
        "message": "TTS 수정 완료"
    }

# ────────────────────────────────────────────────────────────
#  /generate-tts  (custom TTS 생성/수정)
# ────────────────────────────────────────────────────────────
@app.post("/generate-tts")
async def generate_tts_custom(
    text: str = Form(...),           # TTS로 변환할 텍스트
    voice_id: str = Form(...),       # 사용할 Voice ID
    user_id: int = Form(...),        # 사용자 식별자
    tts_id: int = Form(None)         # 수정할 기존 TTS ID (없으면 생성)
):
    """
    user_id, text, voice_id, (tts_id)를 Form 데이터로 받아
    user_files/{user_id} 폴더에 TTS 파일 생성 및 DB 반영 + 파형 생성(waveform_url 저장)
    """
    try:
        # 사용자별 폴더 생성
        user_folder = os.path.join(USER_FILES_FOLDER, str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        # DB 커넥션
        conn = get_connection()
        curs = conn.cursor()

        # 기존 TTS가 있으면 translations 업데이트, 새로 생성시 삽입
        if tts_id:
            curs.execute("SELECT translation_id FROM tts WHERE tts_id = %s;", (tts_id,))
            row = curs.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="tts_id를 찾을 수 없습니다.")
            translation_id = row[0]
            # 번역 텍스트 업데이트
            curs.execute("UPDATE translations SET text = %s WHERE translation_id = %s;", (text, translation_id))
            filename = f"tts_{tts_id}.mp3"
        else:
            # 신규 번역 레코드 삽입
            curs.execute(
                "INSERT INTO translations(text, language) VALUES(%s, %s) RETURNING translation_id;",
                (text, "en")  # 언어코드는 필요에 따라 수정
            )
            translation_id = curs.fetchone()[0]
            # 우선 임시 이름으로 생성 (tts_id를 알기 전)
            filename = f"tts_tmp_{uuid4().hex}.mp3"

        # 파일 경로
        output_path = os.path.join(user_folder, filename)

        # ElevenLabs TTS 생성
        url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}?output_format=mp3_44100_128"
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"text": text, "model_id": "eleven_multilingual_v2"})
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {resp.text}")

        # 파일 저장 및 길이 측정
        with open(output_path, 'wb') as f:
            f.write(resp.content)
        duration = librosa.get_duration(path=output_path)

        # DB tts 테이블 반영 + 파형 생성
        if tts_id:
            # 파형 먼저 생성
            wave_name = f"tts_{tts_id}.png"
            wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
            width_px  = int(max(1, round(float(duration) * 100)))
            generate_waveform_png(output_path, wave_path, width_px=width_px, height_px=100)
            wave_url = f"/waveforms/{wave_name}"

            # tts 업데이트
            curs.execute(
                "UPDATE tts SET voice = %s, duration = %s, file_path = %s, waveform_url = %s WHERE tts_id = %s;",
                (voice_id, duration, output_path, wave_url, tts_id)
            )
        else:
            # tts 신규 삽입하여 tts_id 확보
            curs.execute(
                """
                INSERT INTO tts(translation_id, file_path, voice, start_time, duration)
                VALUES(%s, %s, %s, %s, %s) RETURNING tts_id;
                """,
                (translation_id, output_path, voice_id, 0.0, duration)
            )
            tts_id = curs.fetchone()[0]

            # 파일명을 tts_id 기준으로 정정 (선택 사항)
            new_filename = f"tts_{tts_id}.mp3"
            new_output_path = os.path.join(user_folder, new_filename)
            if new_output_path != output_path:
                try:
                    os.replace(output_path, new_output_path)
                    output_path = new_output_path
                    curs.execute(
                        "UPDATE tts SET file_path = %s WHERE tts_id = %s;",
                        (output_path, tts_id)
                    )
                except Exception:
                    pass  # 파일명 변경 실패해도 치명적이진 않으니 무시

            # 파형 생성 후 URL 저장
            wave_name = f"tts_{tts_id}.png"
            wave_path = os.path.join(WAVEFORM_FOLDER, wave_name)
            width_px  = int(max(1, round(float(duration) * 100)))
            generate_waveform_png(output_path, wave_path, width_px=width_px, height_px=100)
            wave_url = f"/waveforms/{wave_name}"
            curs.execute(
                "UPDATE tts SET waveform_url=%s WHERE tts_id=%s;",
                (wave_url, tts_id)
            )

        conn.commit()
        curs.close()
        conn.close()

        return JSONResponse({"message": "TTS 생성 완료", "tts_id": tts_id, "waveform_url": wave_url})

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"generate-tts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-voice-model")
async def create_voice_model(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: str = Form(None),
    remove_background_noise: bool = Form(False),
    files: List[UploadFile] = File(...),
    image: UploadFile = File(None),           # ← NEW: 이미지 수신 필드
):
    """
    업로드된 오디오 파일을 받아…
    ElevenLabs API로 보이스 모델을 만든 뒤 DB에 저장합니다.
    """
    all_sample_parts = []
    temp_paths = []
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # ── 오디오 파일 처리 ──
    for file in files:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"파일 {file.filename}이 10MB 초과")

        original_name = os.path.splitext(file.filename)[0]
        base_name = (
            original_name[:-len("_audio")]
            if file.filename.endswith("_audio")
            else original_name
        )
        original_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        logging.info(f"저장: {original_path}")
        with open(original_path, "wb") as f:
            f.write(await file.read())
        temp_paths.append(original_path)

        if remove_background_noise:
            # TODO: 배경 소음 제거
            pass

        # Spleeter 분리
        try:
            separator = Separator("spleeter:2stems")
            separator.separate_to_file(original_path, AUDIO_FOLDER)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Spleeter 실패: {e}")

        vocal_path, _ = find_spleeter_vocals(AUDIO_FOLDER, base_name)
        fixed_vocal_path = os.path.join(AUDIO_FOLDER, f"{base_name}_vocals.wav")
        shutil.move(vocal_path, fixed_vocal_path)
        temp_paths.append(fixed_vocal_path)
        shutil.rmtree(os.path.join(AUDIO_FOLDER, base_name), ignore_errors=True)
        shutil.rmtree(os.path.join(AUDIO_FOLDER, f"{base_name}_audio"), ignore_errors=True)

        # 샘플 병합
        merge_dir = os.path.join(AUDIO_FOLDER, f"{base_name}_merged")
        os.makedirs(merge_dir, exist_ok=True)
        temp_paths.append(merge_dir)
        merged_sample = merge_nonsilent_audio_improved(
            fixed_vocal_path, merge_dir,
            output_filename="merged_sample.mp3",
            fade_duration=200
        )

        # 샘플 분할
        split_dir = os.path.join(AUDIO_FOLDER, f"{base_name}_split")
        os.makedirs(split_dir, exist_ok=True)
        temp_paths.append(split_dir)
        parts = split_merged_audio(
            merged_sample, split_dir,
            max_duration_sec=30, max_samples=25
        )
        all_sample_parts.extend(parts)

    if not all_sample_parts:
        raise HTTPException(status_code=500, detail="샘플 생성 실패")

    # ── NEW: 이미지 저장 ──
    image_url = None
    if image:
        ext = os.path.splitext(image.filename)[1] or ".png"
        filename = f"{uuid4().hex}{ext}"
        dest = os.path.join(IMAGE_FOLDER, filename)
        with open(dest, "wb") as img_f:
            img_f.write(await image.read())
        image_url = f"/images/{filename}"

    # ElevenLabs API 호출
    voice_response = create_voice_model_api(
        name=name,
        description=description,
        sample_file_paths=all_sample_parts
    )
    voice_id = voice_response.get("voice_id")
    if not voice_id:
        raise HTTPException(status_code=500, detail="voice_id 반환 누락")

    # ── DB 저장 (image_url 포함) ──
    with get_db_cursor() as curs:
        curs.execute(
            """
            INSERT INTO voice_models
                (voice_id, name, description, image_url)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (voice_id, name, description, image_url),
        )
        inserted_id = curs.fetchone()[0]

    background_tasks.add_task(delete_temp_files, temp_paths)

    return JSONResponse(
        {
            "message": "보이스 모델 생성 완료",
            "db_id": inserted_id,
            "voice_id": voice_id,
            "name": name,
            "description": description,
            "image_url": image_url,    # ← NEW: 반환
        },
        status_code=200
    )


@app.get("/voice-models")
async def list_voice_models():
    with get_db_cursor() as curs:
        curs.execute("""
            SELECT id, name, voice_id, description, image_url
            FROM voice_models
            ORDER BY id;
        """)
        rows = curs.fetchall()

    models = []
    for db_id, name, vid, desc, img in rows:
        models.append({
            "db_id":       db_id,
            "name":        name,
            "voice_id":    vid,
            "description": desc,
            "image_url":   img,       # ← NEW: 포함
        })
    return JSONResponse(content=models)

def generate_waveform_png(audio_path: str, out_path: str, width_px: int, height_px: int = 100):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y) == 0:
        y = np.zeros(22050//2)
    samples_per_px = max(1, len(y) // width_px)
    peaks = []
    for i in range(width_px):
        s = i * samples_per_px
        e = min(len(y), s + samples_per_px)
        amp = float(np.max(np.abs(y[s:e]))) if s < len(y) else 0.0
        peaks.append(amp)
    maxamp = max(peaks) or 1.0
    peaks = [p / maxamp for p in peaks]

    img = Image.new("RGB", (width_px, height_px), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    mid = height_px / 2.0
    for x, p in enumerate(peaks):
        h = p * (height_px * 0.9)
        y0 = int(mid - h/2); y1 = int(mid + h/2)
        draw.line((x, y0, x, y1), fill="#007bff", width=1)
    img.save(out_path, format="PNG")