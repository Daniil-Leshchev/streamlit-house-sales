FROM python:3.11-alpine

WORKDIR /app

RUN apk add --no-cache \
    build-base \
    musl-dev \
    python3-dev \
    libffi-dev \
    openblas-dev \
    linux-headers \
    cmake

COPY requirements.txt ./
RUN pip install --no-cache-dir nvidia-pyindex \
    && pip install --no-cache-dir -r requirements.txt

COPY best_xgb.joblib ./
COPY main.py ./

EXPOSE 8000

ENTRYPOINT ["streamlit", "run"]
CMD ["main.py", "--server.port=8000", "--server.address=0.0.0.0"]