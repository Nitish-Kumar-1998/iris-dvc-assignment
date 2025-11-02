# ===============================
# Dockerfile for Iris FastAPI API
# ===============================

# Step 1: Use a lightweight Python base image
FROM python:3.12-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy all project files
COPY . /app

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn mlflow

# Step 5: Expose port 8080
EXPOSE 8080

# Step 6: Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
