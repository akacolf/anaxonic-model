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

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

CMD ["python", "-m", "anaxonic.train", "--config", "configs/tiny.json", "--data", "data/sample.txt"]
