FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential \
         ffmpeg \
         ttf-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY best_xgb.joblib/ ./
COPY main.py .

EXPOSE 8000

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8000"]