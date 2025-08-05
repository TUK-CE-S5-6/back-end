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

# FastAPI 앱 생성
app = FastAPI()

DB_NAME = "test"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5433"

ELEVENLABS_API_KEY = "eleven-keyf"
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# OpenAI API 설정 (번역용)
OPENAI_API_KEY = "gpt-key"
openai.api_key = OPENAI_API_KEY

# ── NEW: 언어별 평균 읽기 속도 (음절/문자 per sec)
AVG_SYL_PER_SEC = {"ko": 6.2, "en": 6.19, "ja": 7.84, "zh": 5.18}
TOL_RATIO = 0.02
MIN_SENT_LEN = 4
SUPPORTED_LANGS = set(AVG_SYL_PER_SEC)  # {'ko','en','ja','zh'}

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

# ― NEW: 한글 음절/문자 수 세기
def count_units(txt: str, lang: str) -> int:
    if lang == "ko":
        return sum('\uAC00' <= c <= '\uD7A3' for c in txt)
    return len(txt)

# ― NEW: 길이-의식 번역 함수
def translate_length_aware(
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    dur_sec: float,
    voice_id: str,
    db_cursor,
    retries: int = 4,
):
    tgt = tgt_lang[:2]

    cps     = get_cps(db_cursor, voice_id, tgt)      # 보이스-언어 글자/초
    target  = int(dur_sec * cps)                     # 목표 글자 수
    tol     = max(2, round(target * TOL_RATIO))       # ±2% (최소 2글자)
    lo, hi  = target - tol, target + tol
    unit    = "Hangul syllables" if tgt == "ko" else "characters"

    goal = target                                    # ⬅️ ① 첫 목표값
    for _ in range(retries):
        sys_msg = (
            f"Translate from {src_lang[:2]} to {tgt}. "
            f"Make the output **exactly {goal} {unit}** long. "
            "Translate faithfully—KEEP all proper nouns; do NOT summarize."
        )
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user",   "content": src_text}],
        ).choices[0].message.content.strip()

        n = count_units(res, tgt)
        if lo <= n <= hi:           # ✅ 허용 범위 안이면 반환
            return res

        # ⬇️ 길거나 짧으면 목표 글자수를 1씩 조정
        if n > hi:
            goal -= 1
        else:      # n < lo
            goal += 1
        goal = max(MIN_SENT_LEN, goal)   # 최소 길이 보장

    return res   # 마지막 시도라도 반환

# ― Stretch (±3 % 이내)
def stretch_audio(input_path: str, output_path: str, current_duration: float, desired_duration: float) -> float:
    speed = current_duration / desired_duration
    speed = max(0.97, min(1.03, speed))  # ±3 %
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

def adjust_tts_duration(file_path: str, desired_duration: float) -> float:
    y, sr = librosa.load(file_path, sr=None)
    current_duration = librosa.get_duration(y=y, sr=sr)
    rate = current_duration / desired_duration
    rate = min(max(rate, 0.9), 1.1)
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
app.mount("/uploaded_videos", StaticFiles(directory="uploaded_videos"), name="videos")
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
        
def stretch_audio(input_path: str, output_path: str, current_duration: float, desired_duration: float) -> float:
    """
    pydub을 사용해 파일 속도를 조절.
    Returns: 새로 조정된 길이 (초)
    """
    speed = current_duration / desired_duration
    logging.debug(f"→ Applying speed change: {speed:.4f}×")

    sound = AudioSegment.from_file(input_path)
    new_frame_rate = int(sound.frame_rate * speed)
    stretched = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
    stretched = stretched.set_frame_rate(sound.frame_rate)
    ensure_folder(os.path.dirname(output_path))
    stretched.export(output_path, format="mp3", bitrate="192k")

    new_duration = len(stretched) / 1000.0
    logging.debug(f"→ New duration: {new_duration:.3f}s")
    return new_duration

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
#  /generate-tts-from-stt
#     1) 원본 STT → 길이의식 번역
#     2) TTS 생성·검증
#     3) translations  UPSERT
#     4) tts           INSERT 또는 UPDATE(file_path만 최신화)
# ────────────────────────────────────────────────────────────
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
    MAX_RETRY, TOL = 20, 0.05
    tts_dir = os.path.join(AUDIO_FOLDER, f"{video_id}_tts")
    ensure_folder(tts_dir)

    with get_db_cursor() as cur:
        # ① 영상에 딸린 STT 줄 조회
        cur.execute(
            """
            SELECT tr.transcript_id, tr.text, tr.start_time, tr.end_time,
                   tr.speaker,
                   LEFT(p.source_language,2), LEFT(p.target_language,2)
              FROM transcripts tr
              JOIN videos v ON v.video_id = tr.video_id
              JOIN projects p ON p.project_id = v.project_id
             WHERE tr.video_id = %s
             ORDER BY tr.start_time;
            """,
            (video_id,),
        )
        rows = cur.fetchall()

        for tid, src, st, et, spk, src_l, tgt_l in rows:
            dur = float(et) - float(st)
            voice = speaker_voice.get(spk)
            if dur <= 0 or not voice:
                continue

            # ② 길이-맞춤 번역
            text = translate_length_aware(
                src_text=src,
                src_lang=src_l,
                tgt_lang=tgt_l,
                dur_sec=dur,
                voice_id=voice,
                db_cursor=cur,
            )

            # ③ TTS 생성 & 길이 검수
            for n in range(MAX_RETRY):
                tmp_path = os.path.join(tts_dir, f"tmp_{tid}_{n}.mp3")
                synth_to_file(voice, text, tmp_path)

                real = librosa.get_duration(path=tmp_path)
                if abs(real - dur) / dur <= TOL:
                    final_mp3, final_dur = tmp_path, real
                    break

                # 길이 불일치 → 글자 수 비례 재조정
                need_len = max(8, round(len(text) * dur / real))
                sys_msg = (
                    f"Rewrite to {need_len} "
                    f"{'syllables' if tgt_l=='ko' else 'chars'}. Keep names."
                )
                text = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": text},
                    ],
                    timeout=15,
                    request_timeout=15,
                ).choices[0].message.content.strip()
            else:
                # ── fallback: SSML prosody rate 보정으로 최종 TTS 재생성 ──
                rate = dur / real
                ssml = f'<speak><prosody rate="{int(rate*100)}%">{text}</prosody></speak>'
                fallback_path = os.path.join(tts_dir, f"fallback_{tid}.mp3")
                synth_to_file(voice, ssml, fallback_path)
                final_dur = librosa.get_duration(path=fallback_path)
                final_mp3 = fallback_path

            # ④ translations UPSERT
            cur.execute(
                """
                INSERT INTO translations (transcript_id, language, text)
                VALUES (%s, %s, %s)
                ON CONFLICT (transcript_id, language)
                DO UPDATE SET text = EXCLUDED.text
                RETURNING translation_id;
                """,
                (tid, tgt_l, text),
            )
            translation_id = cur.fetchone()[0]

            # ⑤ tts INSERT 또는 UPDATE(경로·길이만 최신화)
            cur.execute(
                "SELECT tts_id FROM tts WHERE translation_id = %s;",
                (translation_id,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    """
                    UPDATE tts
                       SET file_path = %s,
                           voice     = %s,
                           duration  = %s
                     WHERE tts_id  = %s;
                    """,
                    (final_mp3, voice, final_dur, row[0]),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO tts (translation_id, file_path, voice,
                                     start_time, duration)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (translation_id, final_mp3, voice, st, final_dur),
                )

    return {"message": "TTS 완료"}

# ────────────────────────────────────────────────────────────
#  /edit-tts
#     1) 재번역
#     2) 새 mp3 생성·검증
#     3) translations UPSERT
#     4) tts row 덮어쓰기 (file_path 포함)
# ────────────────────────────────────────────────────────────
@app.post("/edit-tts")
async def edit_tts(
    tts_id: int   = Form(...),
    voice: str    = Form(...),
    text: str     = Form(...),
    update_src: bool = Form(True),
):
    MAX_RETRY, TOL = 5, 0.03

    with get_db_cursor() as cur:
        # 기존 정보
        cur.execute(
            """
            SELECT  tts.translation_id, ts.transcript_id,
                    tts.start_time, tts.duration, tts.file_path,
                    LEFT(p.source_language,2), LEFT(p.target_language,2)
              FROM  tts
              JOIN  translations t ON t.translation_id = tts.translation_id
              JOIN  transcripts  ts ON ts.transcript_id = t.transcript_id
              JOIN  videos       v  ON v.video_id       = ts.video_id
              JOIN  projects     p  ON p.project_id     = v.project_id
             WHERE  tts.tts_id = %s;
            """,
            (tts_id,),
        )
        (
            old_tran,
            trans_id,
            st,
            dur,
            old_path,
            src_l,
            tgt_l,
        ) = cur.fetchone()

        base_dir = os.path.dirname(old_path)
        ensure_folder(base_dir)

        # ① 재번역
        new_text = translate_length_aware(
            src_text=text,
            src_lang=src_l,
            tgt_lang=tgt_l,
            dur_sec=dur,
            voice_id=voice,
            db_cursor=cur,
        )

        # ② mp3 재생성
        for n in range(MAX_RETRY):
            tmp_path = os.path.join(base_dir, f"tmp_{tts_id}_{n}.mp3")
            synth_to_file(voice, new_text, tmp_path)

            real = librosa.get_duration(path=tmp_path)
            if abs(real - dur) / dur <= TOL:
                final_mp3, final_dur = tmp_path, real
                break

            need_len = max(8, round(len(new_text) * dur / real))
            sys_msg  = (
                f"Rewrite to {need_len} "
                f"{'syllables' if tgt_l=='ko' else 'chars'}. Keep names."
            )
            new_text = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": new_text},
                ],
            ).choices[0].message.content.strip()
        else:
            final_mp3, final_dur = tmp_path, real

        # ③ translations UPSERT
        cur.execute(
            """
            INSERT INTO translations (transcript_id, language, text)
            VALUES (%s, %s, %s)
            ON CONFLICT (transcript_id, language)
            DO UPDATE SET text = EXCLUDED.text
            RETURNING translation_id;
            """,
            (trans_id, tgt_l, new_text),
        )
        new_tran_id = cur.fetchone()[0]

        # ④ tts 덮어쓰기 (file_path 포함!)
        cur.execute(
            """
            UPDATE tts
               SET translation_id = %s,
                   file_path      = %s,
                   voice          = %s,
                   duration       = %s
             WHERE tts_id         = %s;
            """,
            (new_tran_id, final_mp3, voice, final_dur, tts_id),
        )

        # ⑤ 원본 STT도 바꿀지 여부
        if update_src:
            cur.execute(
                "UPDATE transcripts SET text = %s WHERE transcript_id = %s;",
                (text, trans_id),
            )

    return {
        "tts_id": tts_id,
        "duration": final_dur,
        "file_path": final_mp3,
        "translation_id": new_tran_id,
    }


# ────────────────────────────────────────────────────────────
#  공통: ElevenLabs 호출 래퍼
# ────────────────────────────────────────────────────────────
def synth_to_file(voice_id: str, text: str, out_path: str):
    resp = requests.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
        "?output_format=mp3_44100_128",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"speed": 1.0},
        },
    )
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)


@app.post("/generate-tts")
async def generate_tts_custom(
    text: str = Form(...),           # TTS로 변환할 텍스트
    voice_id: str = Form(...),       # 사용할 Voice ID
    user_id: int = Form(...),        # 사용자 식별자
    tts_id: int = Form(None)         # 수정할 기존 TTS ID (없으면 생성)
):
    """
    user_id, text, voice_id, (tts_id)를 Form 데이터로 받아
    user_files/{user_id} 폴더에 TTS 파일 생성 및 DB 반영
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
            filename = f"tts_{text}.mp3"

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

        # DB tts 테이블 반영
        if tts_id:
            curs.execute(
                "UPDATE tts SET voice = %s, duration = %s, file_path = %s WHERE tts_id = %s;",
                (voice_id, duration, output_path, tts_id)
            )
        else:
            curs.execute(
                "INSERT INTO tts(translation_id, file_path, voice, start_time, duration) VALUES(%s, %s, %s, %s, %s) RETURNING tts_id;",
                (translation_id, output_path, voice_id, 0.0, duration)
            )
            tts_id = curs.fetchone()[0]

        conn.commit()
        curs.close()
        conn.close()

        # URL은 사용하지 않으므로 반환하지 않음
        return JSONResponse({"message": "TTS 생성 완료", "tts_id": tts_id})

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
):
    """
    업로드된 오디오 파일을 받아 Spleeter로 분리 후 샘플을 생성,
    ElevenLabs API로 보이스 모델을 만든 뒤 DB에 저장합니다.
    """
    all_sample_parts = []
    temp_paths = []
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # 파일 처리
    for file in files:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"파일 {file.filename}의 크기가 10MB를 초과합니다.")

        original_name = os.path.splitext(file.filename)[0]
        base_name = original_name[:-len("_audio")] if file.filename.endswith("_audio") else original_name
        original_path = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        logging.info(f"📥 파일 저장 시작: {file.filename}")
        with open(original_path, "wb") as f:
            f.write(await file.read())
        temp_paths.append(original_path)

        if remove_background_noise:
            # TODO: 배경 소음 제거 처리 호출
            pass

        try:
            separator = Separator("spleeter:2stems")
            separator.separate_to_file(original_path, AUDIO_FOLDER)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Spleeter 실행 실패: {e}")

        vocal_path, _ = find_spleeter_vocals(AUDIO_FOLDER, base_name)
        fixed_vocal_path = os.path.join(AUDIO_FOLDER, f"{base_name}_vocals.wav")
        shutil.move(vocal_path, fixed_vocal_path)
        temp_paths.append(fixed_vocal_path)
        shutil.rmtree(os.path.join(AUDIO_FOLDER, base_name), ignore_errors=True)
        shutil.rmtree(os.path.join(AUDIO_FOLDER, f"{base_name}_audio"), ignore_errors=True)

        merge_dir = os.path.join(AUDIO_FOLDER, f"{base_name}_merged")
        os.makedirs(merge_dir, exist_ok=True)
        temp_paths.append(merge_dir)
        merged_sample = merge_nonsilent_audio_improved(
            fixed_vocal_path, merge_dir,
            output_filename="merged_sample.mp3",
            fade_duration=200
        )

        split_dir = os.path.join(AUDIO_FOLDER, f"{base_name}_split")
        os.makedirs(split_dir, exist_ok=True)
        temp_paths.append(split_dir)
        parts = split_merged_audio(
            merged_sample, split_dir,
            max_duration_sec=30, max_samples=25
        )
        all_sample_parts.extend(parts)

    if not all_sample_parts:
        raise HTTPException(status_code=500, detail="샘플 파일 생성 실패")

    voice_response = create_voice_model_api(
        name=name,
        description=description,
        sample_file_paths=all_sample_parts
    )
    voice_id = voice_response.get("voice_id")
    if not voice_id:
        raise HTTPException(status_code=500, detail="보이스 모델 생성 실패: voice_id가 반환되지 않았습니다.")
    logging.info(f"✅ 보이스 모델 생성 완료: {voice_id}")

    # DB 저장
    with get_db_cursor() as curs:
        curs.execute(
            """
            INSERT INTO voice_models (voice_id, name, description)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (voice_id, name, description)
        )
        inserted_id = curs.fetchone()[0]
    logging.info(f"📌 DB에 voice_models 저장 완료 (id: {inserted_id})")

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

    
@app.get("/voice-models")
async def list_voice_models():
    with get_db_cursor() as curs:
        curs.execute("""
            SELECT id, name, voice_id, description
            FROM voice_models
            ORDER BY id;
        """)
        rows = curs.fetchall()
    models = [
        {"db_id": row[0], "name": row[1], "voice_id": row[2], "description": row[3]}
        for row in rows
    ]
    return JSONResponse(content=models)