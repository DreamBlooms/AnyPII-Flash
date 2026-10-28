FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    ANYPPI_FLASH_HOST=0.0.0.0 \
    ANYPPI_FLASH_PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[model]"

EXPOSE 8080

CMD ["anypii-flash"]
