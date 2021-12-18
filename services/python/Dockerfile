FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3
WORKDIR /code
RUN apt update -y && apt upgrade -y
RUN apt install ffmpeg libpq-dev -y
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed pyyaml
RUN wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install torch-1.9.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install ftfy regex tqdm opencv-python simplejpeg vidgear[asyncio] pgvector psycopg2 fastapi[all] uvicorn
RUN pip3 install git+https://github.com/openai/CLIP.git
COPY . .
ENV LANG C.UTF-8
ENV PYTHONIOENCODING utf-8
CMD ["python3", "main.py"]
# CMD ["python3", "-m", "uvicorn", "main:app", "--reload", "--host=0.0.0.0"]
