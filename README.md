# Task 10 – Containerizing Flask Deep Learning API

## Objective
To package a Flask-based deep learning API into a Docker container for local deployment.

## Technologies Used
- Python
- Flask
- TensorFlow / Keras
- MobileNetV2
- Docker
- Docker Desktop

## Project Structure
- app.py – Flask deep learning API
- Dockerfile – Docker image configuration
- requirements.txt – Python dependencies
- .dockerignore – Files excluded from Docker build context

## Docker Image
flask-dl-api:latest

## Optimized Image
flask-dl-api:optimized

## Port
5000

## API Endpoints

### Home
GET /

### Prediction
POST /predict

## Run Docker Container

```bash
docker run -d -p 5000:5000 --name flask-dl-container flask-dl-api:optimized
