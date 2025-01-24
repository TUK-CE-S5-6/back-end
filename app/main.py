import os
import openai
import jaydebeapi
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from whisper import load_model
from moviepy.editor import AudioFileClip
import logging
from elevenlabs import ElevenLabs

#########################
# H2 JDBC 설정 부분
#########################
# H2 서버 모드가 이미 실행 중이라고 가정
# 예) java -jar h2-2.2.220.jar -tcp -tcpPort 9092 -web -webPort 8082 -webAllowOthers
H2_JAR_PATH = os.path.abspath("h2-2.2.220.jar")  # H2 드라이버 JAR 파일 경로
JDBC_URL = "jdbc:h2:tcp://localhost:9092/~/test" # 실제 서버 모드 URL
DRIVER_CLASS = "org.h2.Driver"
DB_USER = "sa"
DB_PASSWORD = "1234"

#########################
# FastAPI 앱 생성
#########################
app = FastAPI()

#########################
# CORS 설정
#########################
# 로컬 클라이언트 http://localhost:3000 에서 요청을 허용하기 위한 예시
origins = [
    "http://localhost:3000",
    # 필요하다면 추가 도메인을 여기에 등록
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
    
# 요청 본문 데이터 모델 정의
class TTSRequest(BaseModel):
    file_name: str
    
    
# 업로드된 동영상 저장 디렉토리
UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 오디오 추출 저장 디렉토리
AUDIO_FOLDER = "extracted_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Whisper 모델 로드 (large 모델을 사용할 경우 메모리 주의)
WHISPER_MODEL = "large"  # small, medium, large로 변경 가능
model = load_model(WHISPER_MODEL)

# OpenAI API 키 설정
OPENAI_API_KEY = "token key"
openai.api_key = OPENAI_API_KEY

#########################
# DB 연결 함수 (JayDeBeAPI)
#########################
def get_connection():
    """
    JayDeBeAPI로 H2 서버 모드에 접속하기 위한 함수
    """
    conn = jaydebeapi.connect(
        DRIVER_CLASS,
        JDBC_URL,
        [DB_USER, DB_PASSWORD],
        H2_JAR_PATH
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
    예시용. users 테이블 생성 + 샘플 데이터 삽입
    user_id (PK, AUTO_INCREMENT), username(VARCHAR(50), UNIQUE), password(VARCHAR(255))
    """
    conn = get_connection()
    curs = conn.cursor()

    # AUTO_INCREMENT 사용을 위해 IDENTITY 컬럼 설정
    curs.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        );
    """)

    # 샘플 데이터 (중복 시 에러 가능, 간단 예시)
    try:
        curs.execute("INSERT INTO users (username, password) VALUES ('Alice', 'alice123');")
        curs.execute("INSERT INTO users (username, password) VALUES ('Bob', 'bob123');")
    except:
        pass

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
    회원가입 후 username과 마지막 user_id를 반환 (안전한 환경에서만 사용).
    """
    conn = get_connection()
    curs = conn.cursor()

    # 중복 username 체크
    curs.execute("SELECT user_id FROM users WHERE username = ?;", (data.username,))
    row = curs.fetchone()
    if row:
        curs.close()
        conn.close()
        raise HTTPException(status_code=400, detail="이미 사용 중인 사용자 이름입니다.")

    # 사용자 삽입
    curs.execute("INSERT INTO users (username, password) VALUES (?, ?);", (data.username, data.password))
    conn.commit()

    # 테이블에서 가장 큰 user_id를 가져오기 (가장 최근 삽입된 행의 ID)
    curs.execute("SELECT MAX(user_id) FROM users;")
    new_id_row = curs.fetchone()
    new_id = new_id_row[0] if new_id_row else None

    curs.close()
    conn.close()

    # user_id와 username 반환
    return {"user_id": new_id, "username": data.username}



#########################
# 사용자 목록 조회 (Optional)
#########################
@app.get("/users", response_model=list[User])
def list_users():
    """
    DB에 저장된 모든 사용자 목록을 반환
    """
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id, username FROM users;")
    rows = curs.fetchall()
    curs.close()
    conn.close()

    return [{"user_id": r[0], "username": r[1]} for r in rows]

#########################
# 특정 사용자 조회 (Optional)
#########################
@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int):
    """
    user_id에 해당하는 사용자 정보 조회
    """
    conn = get_connection()
    curs = conn.cursor()
    curs.execute("SELECT user_id, username FROM users WHERE user_id = ?;", (user_id,))
    row = curs.fetchone()
    curs.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"user_id": row[0], "username": row[1]}

#########################
# 동영상 업로드 라우트 추가
#########################
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    클라이언트에서 동영상을 업로드하는 엔드포인트
    """
    try:
        # 업로드된 파일 저장 경로 설정
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return JSONResponse(content={"message": f"파일 {file.filename} 업로드 완료"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.post("/stt-video")
async def upload_and_transcribe(file: UploadFile = File(...)):
    """
    동영상을 업로드받아 Whisper로 STT 수행 후 ChatGPT로 번역
    """
    try:
        logging.info("STT 및 번역 작업 시작")

        # 동영상 저장
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"동영상 저장 완료: {video_path}")

        # 동영상에서 오디오 추출
        audio_path = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(file.filename)[0]}.wav")
        audio_clip = AudioFileClip(video_path)
        audio_clip.write_audiofile(audio_path)
        audio_clip.close()
        logging.info(f"오디오 추출 완료: {audio_path}")

        # Whisper를 사용해 텍스트 변환 수행 (타임스탬프 포함)
        logging.info("Whisper 대본 생성 시작")
        result = model.transcribe(audio_path, word_timestamps=True)
        logging.info("Whisper 대본 생성 완료")

        # 한글 대본 생성 및 저장
        transcription_file = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(file.filename)[0]}_transcription.txt")
        logging.info(f"한글 대본 저장 시작: {transcription_file}")
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write("[한글번역]\n")
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {text}\n")
        logging.info(f"한글 대본 저장 완료: {transcription_file}")

        # 영어 번역 수행 및 저장
        logging.info("ChatGPT 번역 시작")
        english_transcription_file = os.path.join(AUDIO_FOLDER, f"{os.path.splitext(file.filename)[0]}_translation.txt")
        with open(english_transcription_file, "w", encoding="utf-8") as f:
            f.write("[영어번역]\n")
            for segment in result["segments"]:
                text = segment["text"]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Translate the following Korean text into English. Keep it concise and accurate."},
                            {"role": "user", "content": text},
                        ]
                    )
                    translated_text = response["choices"][0]["message"]["content"].strip()
                    start_time = segment["start"]
                    end_time = segment["end"]
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {translated_text}\n")
                except Exception as e:
                    logging.error(f"번역 실패: {str(e)}")
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] 번역 실패: {text}\n")
        logging.info(f"영어 번역 저장 완료: {english_transcription_file}")

        # 결과 반환
        logging.info("결과 반환 시작")
        return JSONResponse(content={
            "transcription": open(transcription_file, "r", encoding="utf-8").read(),
            "translation": open(english_transcription_file, "r", encoding="utf-8").read()
        }, status_code=200)

    except Exception as e:
        logging.error(f"STT 또는 번역 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT 또는 번역 실패: {str(e)}")

@app.post("/generate-tts")
async def generate_tts(request: TTSRequest):
    """
    저장된 번역 메모장을 읽어서 영어 음성 생성
    """
    try:
        logging.info("TTS 생성 시작")

        file_name = request.file_name  # 파일 이름 가져오기
        english_transcription_file = os.path.join(AUDIO_FOLDER, f"{file_name}_translation.txt")
        tts_output_dir = os.path.join(AUDIO_FOLDER, f"{file_name}_tts")
        os.makedirs(tts_output_dir, exist_ok=True)

        if not os.path.exists(english_transcription_file):
            raise FileNotFoundError(f"번역 파일을 찾을 수 없습니다: {english_transcription_file}")

        # 번역 파일 읽기
        with open(english_transcription_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ElevenLabs 클라이언트 초기화
        elevenlabs_client = ElevenLabs(api_key="token key")  # API 키 설정

        for i, line in enumerate(lines):
            # 타임스탬프가 있고 "-"가 포함된 줄만 처리
            if line.startswith("[") and "-" in line and "]" in line:
                try:
                    # 타임스탬프 뒤의 텍스트 추출
                    content = line.split("]", 1)[1].strip()

                    # TTS 생성
                    audio_generator = elevenlabs_client.text_to_speech.convert(
                        voice_id="5Af3x6nAIWjF6agOOtOz",
                        model_id="eleven_multilingual_v2",
                        text=content,
                    )

                    # TTS 파일 저장
                    tts_audio_path = os.path.join(tts_output_dir, f"segment_{i}.mp3")
                    with open(tts_audio_path, "wb") as tts_file:
                        for chunk in audio_generator:
                            tts_file.write(chunk)
                    logging.info(f"TTS 생성 완료: {tts_audio_path}")
                except Exception as e:
                    logging.error(f"TTS 생성 실패 (line {i+1}): {str(e)}")

        logging.info("TTS 생성 완료")
        return JSONResponse(
            content={
                "message": "TTS 생성이 완료되었습니다.",
                "files": [os.path.join(tts_output_dir, f) for f in os.listdir(tts_output_dir)]
            },
            status_code=200
        )

    except Exception as e:
        logging.error(f"TTS 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {str(e)}")