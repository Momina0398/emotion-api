FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# CPU-only torch keeps the image much smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# Default is the dummy model; mount your checkpoint and set these to use the real one:
#   docker run -p 8000:8000 -e MODEL_BACKEND=madx -e ADAPTER_PATH=/adapter -v /path/to/rus_track_a_C1:/adapter emotion-api
ENV MODEL_BACKEND=dummy
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
