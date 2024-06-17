FROM python:3.11-slim

# Arguments for user and group ID
ARG UID=1000
ARG GID=1000

# Environment variables for Python and Poetry
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y build-essential curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry --version

# Set up the work directory
WORKDIR /app

# Create a non-root user
RUN groupadd -g ${GID} -r web && \
    useradd -u ${UID} -r -g web -m -d /app web && \
    mkdir -p /var/cache/pypoetry && \
    chown -R web:web /var/cache/pypoetry && \
    chown -R web:web /app

# Copy Poetry files and install dependencies as root
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-interaction --no-ansi

# Change ownership of relevant directories
RUN chown -R web:web /usr/local/lib/python3.11/site-packages && \
    chown -R web:web /usr/local/bin && \
    chown -R web:web /app

# Switch to the non-root user
USER web

# Copy the application code
COPY --chown=web:web . /app/

# Entry point for the container
CMD ["poetry", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
