# AI Wellness Recommendation System

Personalized health insights using real Fitbit data.

## Features
- Mood Score prediction using Random Forest & CatBoost
- Productivity Score calculation
- Indian food recommendation based on mood & activity
- Heart rate variability analysis

## Dataset
- Fitbit Public Dataset (Kaggle)
- 33 users, 31 days

## Setup
pip install -r requirements.txt
python create_unified_dataset.py
streamlit run app.py

## Models
- Random Forest — R² 0.63
- CatBoost — R² 0.63
