# Email Marketing Success Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/SciKit--Learn-1.5.0-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project to predict customer engagement in email marketing campaigns. This application predicts whether a customer is likely to click on an email based on their profile and historical interaction data.

## 🚀 Live Demo

You can try the live application hosted on Hugging Face Spaces:

**[https://huggingface.co/spaces/my-ml-experiments/my-app](https://huggingface.co/spaces/my-ml-experiments/my-app)**

![App Demo](https://i.imgur.com/s4n4V8f.png) 
*(Suggestion: You can replace this static image with a GIF showcasing your app's functionality for a more dynamic presentation.)*

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [How to Run Locally](#-how-to-run-locally)
- [Dataset](#-dataset)

## 🎯 Project Overview

In the world of digital marketing, understanding customer behavior is key to success. Email campaigns are a vital tool, but their effectiveness can vary greatly. This project tackles this challenge by building a machine learning model that predicts whether a user will click on a marketing email.

By predicting user engagement, businesses can:
- **Optimize Campaign Strategy:** Focus resources on customer segments with a higher likelihood of interaction.
- **Personalize Content:** Tailor email content to user profiles that are predicted to engage.
- **Increase ROI:** Improve conversion rates and the overall return on investment for marketing efforts.

This repository contains the complete workflow, from data exploration and model training in Jupyter Notebooks to a deployed, interactive web application.

## ✨ Features

- **Interactive Web Interface:** A user-friendly front-end built with Streamlit to get real-time predictions.
- **Machine Learning Model:** A robust classification model trained to predict email clicks with high accuracy.
- **End-to-End Pipeline:** Covers all stages of a data science project: data cleaning, preprocessing, feature engineering, model training, and deployment.
- **Containerized Application:** Dockerized for easy setup, consistent environments, and scalable deployment.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Web Framework:** Streamlit
- **Containerization:** Docker
- **IDE & Notebooks:** Jupyter Notebook, VS Code
- **Deployment:** Hugging Face Spaces

## 📁 Project Structure

The repository is organized to maintain a clean and scalable workflow.

```
Email-classification-project/
│
├── NoteBooks/
│   └── EDA.ipynb                # Exploratory Data Analysis & Model Experimentation
│
├── src/
│   ├── __init__.py
│   ├── app.py                   # Main Streamlit application file
│   ├── model.pkl                # Pre-trained classification model
│   └── scaler.pkl               # Pre-trained feature scaler
│
├── .gitignore                   # Files to be ignored by Git
├── Dockerfile                   # Instructions for building the Docker image
├── Email_Marketing_Campaign_Dataset_Rounded.xlsx  # The dataset
├── requirements.txt             # Project dependencies
└── README.md                    # You are here!
```

## 🔬 Methodology

The project follows a standard machine learning pipeline:

1.  **Data Understanding:** The initial phase involved exploring the dataset to understand the relationships between different customer attributes and their engagement levels.
2.  **Exploratory Data Analysis (EDA):** Visualizations were created to identify patterns, correlations, and outliers in the data. Key insights were drawn from customer age, purchase history, and previous email interactions.
3.  **Data Preprocessing:**
    -   Handled missing or invalid values (e.g., negative `Time_Spent_On_Website`).
    -   Converted the multi-value `Emails_Clicked` target variable into a binary format (0 for no clicks, 1 for one or more clicks) to frame it as a classification problem.
    -   Applied standard scaling to numerical features to ensure the model treats all features equally.
4.  **Model Building & Evaluation:** Several classification algorithms were tested. A **Random Forest Classifier** was chosen for its strong performance on this tabular dataset. The model was trained and evaluated on metrics like Accuracy, Precision, Recall, and F1-Score.
5.  **Application Development:** A Streamlit application was built to provide a simple interface for users to input customer data and receive a prediction. The trained model and scaler were saved as `.pkl` files to be loaded by the app.

## ⚙️ How to Run Locally

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   [Git](https://git-scm.com/downloads)
-   [Python 3.9+](https://www.python.org/downloads/)
-   [Docker](https://www.docker.com/products/docker-desktop/) (Optional, for containerized setup)

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-github-username>/Email-classification-project.git
cd Email-classification-project
```

### 2. Set Up a Virtual Environment and Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run src/app.py
```

Open your web browser and navigate to `http://localhost:8501`.

### 4. Run with Docker (Alternative)

If you have Docker installed, you can build and run the application in a container.

```bash
# Build the Docker image
docker build -t email-classifier-app .

# Run the Docker container
docker run -p 8501:8501 email-classifier-app
```

Open your web browser and navigate to `http://localhost:8501`.

## 📊 Dataset

The dataset used for this project is `Email_Marketing_Campaign_Dataset_Rounded.xlsx`. It contains customer attributes and their interactions with email campaigns.

**Key Columns:**

-   **Customer_Age:** Age of the customer.
-   **Emails_Opened:** Number of emails opened by the customer.
-   **Emails_Clicked:** Number of links clicked in the emails. **(This is the base for our target variable)**.
-   **Purchase_History:** Total purchase amount by the customer.
-   **Time_Spent_On_Website:** Time in minutes spent on the website.
-   **Days_Since_Last_Open:** Number of days since the customer last opened an email.
-   **Customer_Engagement_Score:** A calculated score for customer engagement.
-   **Opened_Previous_Emails:** (Binary) Whether the customer opened emails previously.
-   **Clicked_Previous_Emails:** (Binary) Whether the customer clicked emails previously.
-   **Device_Type:** (Binary) The device used (e.g., 0 for Desktop, 1 for Mobile).

---
