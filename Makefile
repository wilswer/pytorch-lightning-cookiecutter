# Makefile

# Set the name of your virtual environment
VENV = venv

# Define commands
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Define targets
.PHONY: install clean

# Create the virtual environment
$(VENV):
    python{{ cookiecutter.python_major_version }}.{{ cookiecutter.python_minor_version }} -m venv $(VENV)

# Install requirements using pip
install: $(VENV)
    $(PIP) install -e .

# Clean up the virtual environment
clean:
    rm -r $(VENV)
