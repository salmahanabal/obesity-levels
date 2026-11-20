FROM python:3.11-slim AS build
WORKDIR /app
COPY pyproject.toml config.yaml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /install /usr/local
COPY --chown=app:app config.yaml ./
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 MODEL_SOURCE=joblib
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "obesity.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
