import os
import time
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from elevenlabs.client import ElevenLabs
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 정적 파일 제공: extracted_audio 폴더 전체 제공
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

# API 키와 ElevenLabs 클라이언트 초기화
ELEVENLABS_API_KEY = "eleven-key"
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# 오디오 파일 저장 폴더: extracted_audio 폴더 하위의 sound_effects 폴더 생성
SOUND_EFFECTS_FOLDER = os.path.join("extracted_audio", "sound_effects")
os.makedirs(SOUND_EFFECTS_FOLDER, exist_ok=True)

def generate_sound_effect(text: str, output_path: str):
    print("Generating sound effects...")
    result = elevenlabs.text_to_sound_effects.convert(
        text=text,
        duration_seconds=10,      # 원하는 사운드 이펙트 길이 (초)
        prompt_influence=0.3,      # 프롬프트가 결과에 미치는 영향 (기본값 0.3)
    )
    with open(output_path, "wb") as f:
        for chunk in result:
            f.write(chunk)
    print(f"Audio saved to {output_path}")
    return output_path

@app.post("/generate-sound-effect")
async def create_sound_effect(text: str = Form(...)):
    try:
        timestamp = int(time.time())
        output_filename = f"sound_effect_{timestamp}.mp3"
        output_path = os.path.join(SOUND_EFFECTS_FOLDER, output_filename)
        generate_sound_effect(text, output_path)
        # 상대 경로 "/extracted_audio/sound_effects/xxx.mp3"를 반환하는 대신 전체 URL 구성
        file_url = f"http://localhost:8002/extracted_audio/sound_effects/{output_filename}"
        return JSONResponse(content={"message": "Sound effect generated successfully", "file_url": file_url}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
