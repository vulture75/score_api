# Use the official Python base image (choose an appropriate Python version)
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code
COPY . .

# Expose the port that FastAPI will run on (adjust the port number as needed)
EXPOSE 9999

# Command to start the FastAPI application
CMD ["python", "app.py"]