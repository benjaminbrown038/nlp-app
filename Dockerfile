FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
ENABLE_TRANSFORMERS=0


WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Optional: copy and install transformers at build time
# COPY requirements-optional.txt ./
# RUN pip install --no-cache-dir -r requirements-optional.txt


COPY app ./app
COPY scripts ./scripts
COPY . .


EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]