FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data/sample.txt ./data/sample.txt
COPY data/prompt_eval.jsonl ./data/prompt_eval.jsonl

# Copy model checkpoint if present (optional — mount at runtime via volume)
COPY runs ./runs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[serve]"

EXPOSE 8000

# Default: serve John Sakura 1.0 over HTTP
# Override CMD to run training: docker run ... python -m anaxonic.train --config configs/tiny.json
CMD ["python", "-m", "anaxonic.serve", "--host", "0.0.0.0", "--port", "8000", "--generate-unknown"]
