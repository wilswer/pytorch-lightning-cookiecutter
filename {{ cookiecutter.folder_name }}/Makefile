# Makefile

# Set the name of your virtual environment
VENV = venv

# Define commands
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
LTT = $(VENV)/bin/ltt

# Define targets
.PHONY: install clean

# Create the virtual environment
$(VENV):
	python{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }} -m venv $(VENV)

# Install requirements using pip
install: $(VENV)
	$(PIP) install light-the-torch
	$(LTT) install torch
	$(PIP) install -e .

# Install development requirements using pip
dev-install: $(VENV)
	$(PIP) install light-the-torch
	$(LTT) install torch
	$(PIP) install -e ".[dev]"

# Clean up the virtual environment
clean:
	rm -r $(VENV)
