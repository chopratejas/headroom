FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN pip install -e .[proxy]
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8787/health || exit 1
ENTRYPOINT ["headroom", "proxy"]