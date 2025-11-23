install:
	pip install -r requirements.txt

download-data:
	python data/scripts/download_dataset.py

train:
	python scripts/train_all_models.py

select-model:
	python scripts/select_best_model.py

optimize:
	python scripts/optimize_model.py

deploy-infra:
	cd terraform && terraform apply

run-app:
	streamlit run streamlit_app/app.py

test:
	pytest tests/

docker-build-lambda:
	docker build -t anomaly-detection-lambda inference/lambda_function/

docker-build-streamlit:
	docker build -t anomaly-detection-app streamlit_app/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
