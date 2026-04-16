FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv pip install --system \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    "uvicorn[standard]>=0.30" fastapi surrealdb httpx meilisearch python-dotenv pydantic pydantic-settings structlog opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp aiocache

# 注意：代码通过卷挂载提供，不需要 COPY
# 开发模式：docker-compose.yml 中挂载 ./wrapper/src:/app/wrapper/src

ENV WRAPPER_PORT=18008 \
    WRAPPER_HOST=0.0.0.0

EXPOSE 18008

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:18008/health || exit 1

CMD ["python", "-m", "wrapper.src.main"]
