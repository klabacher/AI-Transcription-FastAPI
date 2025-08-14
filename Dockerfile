# --- Build Stage ---
# This stage installs all Python dependencies into a virtual environment.
FROM python:3.11-slim as builder

# Set the working directory
WORKDIR /app

# Install poetry for dependency management
# Using poetry is a best practice, but for this project, we stick to requirements.txt
# RUN pip install poetry
# COPY poetry.lock pyproject.toml /app/
# RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# This stage builds the final, slim image for production.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the PATH environment variable to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY . .

# Expose the port the API will run on
EXPOSE 8000

# The default command to run the Uvicorn server for the API.
# The worker service in docker-compose will override this.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
