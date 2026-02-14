.PHONY: install download preprocess train predict app test all clean

install:
	pip install -e ".[dev]"

download:
	python scripts/download_data.py

preprocess:
	python scripts/preprocess.py

train:
	python scripts/train.py

predict:
	python scripts/predict.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

all: download preprocess train predict

clean:
	rm -rf data/processed/*.csv models/*.keras models/*.joblib __pycache__
