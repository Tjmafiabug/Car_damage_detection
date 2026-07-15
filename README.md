<div align="center">

# 🚗🔍 Car Damage Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Learning Project](https://img.shields.io/badge/Learning_Project-6d28d9?style=for-the-badge)

**Detect car damage from images using classical computer vision + a trained ML classifier.**

</div>

---

## About
A hands-on learning project in **computer vision and machine learning**. Instead of
reaching for a heavy deep-learning model, it uses classical CV (edge + contour analysis)
to turn a car image into features, then trains a simple classifier on them — a great way
to understand what an ML pipeline actually does end to end.

## What it does
- **Canny edge detection** to surface scratches and dents
- **Contour counting** to estimate damage severity
- **Logistic Regression** classifier trained on the extracted features
- Stores the dataset + results in **CSV and SQLite**

## How it works
```
image ──► OpenCV (Canny edges + contours) ──► features ──► Logistic Regression ──► damaged? / severity
                                                   │
                                            CSV + SQLite store
```

## Run it
```bash
pip install opencv-python scikit-learn pandas
python main.py
```

## What I learned
- Building an **image → features → model** pipeline from scratch
- When classical CV is enough vs. when you'd reach for deep learning
- Persisting a dataset to CSV + a real SQLite database

<div align="center"><sub>A learning project by <b>Tanishq Jain</b></sub></div>
