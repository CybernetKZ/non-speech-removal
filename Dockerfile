FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app


RUN apt-get update -y && apt-get install -y \
    python3-pip \
    curl \
    ffmpeg \
    ca-certificates \
    gnupg2 \
    git \
 && rm -rf /var/lib/apt/lists/*


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 || true


RUN apt-get update -y && apt-get install -y python3-pip

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/* .

EXPOSE 8000

ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16

CMD ["python3", "main.py"]
