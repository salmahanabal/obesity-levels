PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m obesity.data

eda:
	$(PY) -m obesity.eda

train-clf:
	$(PY) -m obesity.train_classification

train-reg:
	$(PY) -m obesity.train_regression

consistency:
	$(PY) -m obesity.consistency

evaluate:
	$(PY) -m obesity.evaluate

export:
	$(PY) -m obesity.registry

serve:
	uvicorn obesity.api.main:app --host 0.0.0.0 --port 8000

report:
	$(PY) reports/build_report.py

ui:
	streamlit run app/streamlit_app.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest

lint:
	ruff check .

docker-build:
	docker build -f docker/Dockerfile -t obesity-levels .

all: data train-clf train-reg consistency evaluate export

.PHONY: report ui setup data eda train-clf train-reg consistency evaluate export serve mlflow-ui test lint docker-build all
