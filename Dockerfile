FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY app /app/app
COPY deployment /app/deployment
COPY scripts /app/scripts
COPY models /app/models

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
