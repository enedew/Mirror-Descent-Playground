# Use the official Alpine-based Python image
# Use the official Python slim (Debian-based) image
FROM python:3.13.3-slim

# Set environment variables for non-buffered output
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app listens on (Dash apps typically use 8050)
EXPOSE 8050

# Command to run your app using Gunicorn (make sure your app.py exposes "server = app.server")
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "1"]
