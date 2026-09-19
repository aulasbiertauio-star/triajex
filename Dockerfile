FROM python:3.11-slim AS builder
WORKDIR /opt/app
ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim
WORKDIR /opt/app
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY . .
RUN mkdir -p /opt/app/modelos && chown -R appuser:appuser /opt/app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]