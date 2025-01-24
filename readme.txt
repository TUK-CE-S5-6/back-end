venv 가상환경 접속후 DB 실행 터미널 1개, 서버 실행 터미널 1개 총 두개의 터미널이 열려야함.

h2 DB 실행:
java -jar h2-2.2.220.jar -tcp -tcpPort 9092 -web -webPort 8082

서버 실행:
uvicorn app.main:app --reload --port 8000

설치된 pip 정리
pip freeze > requirements.txt

가상환경 pip 설치
pip install -r requirements.txt
