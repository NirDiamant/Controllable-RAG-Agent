# Use an official Python runtime as a parent image
FROM python:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables
# Make sure to set your environment variables here if needed
# ENV API_KEY=your_api_key_here

# Command to run the application
CMD ["streamlit", "run", "simulate_agent.py"]
