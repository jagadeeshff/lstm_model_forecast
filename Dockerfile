# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY excels /app/excels
COPY models /app/models

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 80 to allow communication to/from server (if applicable)
EXPOSE 80

# Define environment variable (optional)
ENV NAME Wind_Speed_Forecast

# Run the application
CMD ["python", "main.py"]
