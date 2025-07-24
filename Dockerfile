# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.8.2 /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

ENV CONTAINER=YES

CMD ["uv", "run", "main.py"]
