FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Allow entrypoint to run
RUN chmod +x docker_entrypoint.sh

EXPOSE 8000
CMD ["bash", "docker_entrypoint.sh"]
