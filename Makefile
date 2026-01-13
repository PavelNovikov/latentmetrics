install:
	uv pip install -e .[test,dev]

prepare:
	black . 
	pytest tests
