.PHONY:
	install test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest --cov=./forecasting

lint:
	pylint forecasting

pre-commit:
	black forecasting	
	isort forecasting