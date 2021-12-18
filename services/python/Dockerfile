FROM python:3.10
WORKDIR /code
RUN apt update -y && apt upgrade -y
RUN apt install ffmpeg libpq-dev -y
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed pyyaml
RUN pip3 install ftfy regex tqdm opencv-python simplejpeg vidgear[asyncio] pgvector psycopg2 fastapi[all] uvicorn torch torchvision
RUN pip3 install git+https://github.com/openai/CLIP.git
COPY . .
ENV LANG C.UTF-8
ENV PYTHONIOENCODING utf-8
CMD ["python3", "main.py"]
# CMD ["python3", "-m", "uvicorn", "main:app", "--reload", "--host=0.0.0.0"]
