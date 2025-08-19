PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

fetch:
	$(PY) data_fetch.py

train:
	$(PY) train_hgb.py

predict:
	$(PY) predict.py $(ARGS)

api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run streamlit_app.py

tune:
	$(PY) tune_hgb.py --trials 30 --holdout 2024

docker-build:
	docker build -t nbaml:latest .

docker-run:
	docker run -p 8000:8000 nbaml:latest
git 