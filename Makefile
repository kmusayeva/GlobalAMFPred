# Target to install dependencies from requirements.txt
install:
	pip install -r requirements.txt


# Target to clean up temporary files and __pycache__ directories
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache

# Target to run tests (assuming you have a tests directory)
#test:
#	pytest tests/

# Target to format code using black (assuming you have black installed)
format:
	black .

# Target to lint code using flake8 (assuming you have flake8 installed)
lint:
	flake8 -v --statistics --max-line-length=100 --disable-noqa .

# Target to check types using mypy (assuming you have mypy installed)
typecheck:
	mypy .

# Phony targets (targets that are not files)
.PHONY: install clean format lint typecheck
