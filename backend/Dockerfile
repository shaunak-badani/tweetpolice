# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY src/requirements.txt .

# Install required build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ python3-dev build-essential cmake make wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Delete installed packages
RUN apt-get remove -y gcc g++ python3-dev build-essential cmake make wget unzip && \
    apt-get autoremove -y

# Copy the rest of the application code to the container
COPY . .

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

WORKDIR /app/src

# Define the command to run the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]