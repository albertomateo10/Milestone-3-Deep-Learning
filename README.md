# Milestone-3-Deep-Learning

This repository contains the code and resources for Milestone 3 Deep Learning, where we have developed and trained a neural network model for detecting people in images. The project involves a custom-trained model and a Dockerized web interface that allows users to upload images and visualize detection predictions.

## Train
This folder contains Jupyter notebooks for both training and testing the neural network model. It includes all the steps performed, from data preprocessing to model evaluation.

##Steps to build and run the dockerized web application:

1. Download/Clone the Repository
2. Prepare the Model. Create a directory named "model" in the project root and place the downloaded model file inside it.
3. Build the Docker Image. Use the provided Dockerfile to build the Docker image: 'docker build -t mi-app:latest .'
4. Run the Docker Container. Run the container, binding port 5000 inside the container to port 5000 on your host machine.
5. Access the Application. Open your browser and go to: http://localhost:5000

### Dataset
https://www.kaggle.com/datasets/fareselmenshawii/human-dataset
### Model
https://drive.google.com/file/d/1kkH4jRSTbzGzWd8nTgQRQ66-EUP4l-LX/view?usp=sharing
