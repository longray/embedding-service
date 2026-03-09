FROM python:3.11-slim as builder

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency file
COPY requirements.txt ./

# Install dependencies using uv toolchain
RUN uv pip install --system -r requirements.txt

###############################
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src ./src

# Expose port
EXPOSE 3001

# Environment defaults (overridable at runtime)
ENV WRAPPER_PORT=3001
ENV WRAPPER_EMBEDDING_SERVICE_URL=http://embedding:18000
ENV WRAPPER_LLM_SERVICE_URL=http://llm:18001
ENV WRAPPER_SURREALDB_URL=ws://surrealdb:8000/rpc

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Start command (use uv to run, per project conventions)
CMD ["uv", "run", "python", "-m", "src.main"]
