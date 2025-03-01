import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set dataset paths:
damaged_path = r"/Users/tanishqjain/Desktop/pyhton project/data1a/training/00-damage"
whole_path = r"/Users/tanishqjain/Desktop/pyhton project/data1a/training/01-whole"

def apply_canny_edge_detection(image_path):
    """Reads an image, converts it to grayscale, and applies Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error Loading Image: {image_path}")
        return None
    edges = cv2.Canny(img, 100, 200)
    return edges

def count_edges(image_path):
    """Counts white pixels (edges) in the image"""
    edges = apply_canny_edge_detection(image_path)
    if edges is None:
        return None
    return np.sum(edges == 255)

def count_contours(image_path):
    """Counts the number of damage contours in the image"""
    edges = apply_canny_edge_detection(image_path)
    if edges is None:
        return None
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def load_images():
    """Loads images and stores them in a Pandas DataFrame"""
    damaged_images = os.listdir(damaged_path)
    whole_images = os.listdir(whole_path)

    print(f"Damaged Car Images: {len(damaged_images)}")
    print(f"Whole Car Images: {len(whole_images)}")

    # Creating DataFrames
    df_damaged = pd.DataFrame(damaged_images, columns=["Image_Name"])
    df_damaged["Image_Path"] = df_damaged["Image_Name"].apply(lambda x: os.path.join(damaged_path, x))
    df_damaged["Label"] = 1  # Damaged cars

    df_whole = pd.DataFrame(whole_images, columns=["Image_Name"])
    df_whole["Image_Path"] = df_whole["Image_Name"].apply(lambda x: os.path.join(whole_path, x))
    df_whole["Label"] = 0  # Whole cars

    # Combine & Shuffle Dataset
    df = pd.concat([df_damaged, df_whole], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def extract_features(df):
    """Applies Edge & Contour feature extraction to DataFrame"""
    df["Edge_Count"] = df["Image_Path"].apply(count_edges)
    df["Contour_Count"] = df["Image_Path"].apply(count_contours)
    return df

def save_data(df):
    """Saves DataFrame to CSV & SQL Database"""
    df.to_csv("car_damage.csv", index=False)
    print("✅ Data saved to car_damage.csv successfully!")

    # Connect to SQLite database
    conn = sqlite3.connect("car_damage.db")
    cursor = conn.cursor()

    # Drop and Create Table
    cursor.execute("DROP TABLE IF EXISTS car_damage;")
    cursor.execute('''
        CREATE TABLE car_damage (
            Image_Name TEXT,
            Label INTEGER,
            Edge_Count INTEGER,
            Contour_Count INTEGER
        )
    ''')

    # Insert data into SQL
    df.to_sql("car_damage", conn, if_exists="replace", index=False)
    print("✅ Data successfully stored in car_damage.db!")

    conn.close()

def load_data():
    """Loads structured data from SQL for AI Training"""
    conn = sqlite3.connect("car_damage.db")
    df = pd.read_sql("SELECT Label, Edge_Count, Contour_Count FROM car_damage", conn)
    conn.close()
    return df


def train_ai_model():
    """Trains Logistic Regression Model for Damage Detection."""
    df = load_data() #Load structured Data
    #define features (X) & labels (Y)
    X = df[["Edge_Count","Contour_Count"]]
    y = df["Label"]
    # Split into training

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Standardise the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate Model Performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))


df = load_images()
df = extract_features(df)
save_data(df)
train_ai_model()
