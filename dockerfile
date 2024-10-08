FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required by some Python packages)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Ensure Rust and Cargo are available in the path
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Command to run the application
CMD ["streamlit", "run", "simulate_agent.py"]