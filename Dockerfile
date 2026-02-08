FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-eng \
       curl \
       ca-certificates \
       libgl1 \
       libglib2.0-0 \
       aptitude \
       htop \
    && TESSDATA_DIR="/usr/share/tesseract-ocr/5/tessdata" \
    && mkdir -p "${TESSDATA_DIR}" \
    && curl -fL -o "${TESSDATA_DIR}/ocrb.traineddata" \
       https://github.com/Shreeshrii/tessdata_ocrb/raw/master/ocrb.traineddata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "presidio_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
