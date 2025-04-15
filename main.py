import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
import os

st.set_page_config(page_title="Workout Recommender", layout="centered")

# ---------- Load or Create users.csv ----------
def load_users():
    if not os.path.exists("users.csv"):
        pd.DataFrame(columns=["username", "password"]).to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password):
    users = load_users()
    if username in users['username'].values:
        return False
    users = pd.concat([users, pd.DataFrame([[username, password]], columns=["username", "password"])])
    users.to_csv("users.csv", index=False)
    return True

# ---------- Authentication ----------
def authenticate_user(username, password):
    users = load_users()
    return not users[(users["username"] == username) & (users["password"] == password)].empty

# ---------- Load and Train Model ----------
@st.cache_resource
def train_model():
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 60, 300),
        'weight': np.random.randint(50, 100, 300),
        'height': np.random.randint(150, 200, 300),
        'steps_walked': np.concatenate([
            np.random.randint(8000, 15000, 100),  # Cardio
            np.random.randint(3000, 7000, 100),   # Yoga
            np.random.randint(1000, 5000, 100)    # Strength
        ]),
        'workout_duration': np.concatenate([
            np.random.randint(30, 90, 100),      # Cardio
            np.random.randint(20, 60, 100),      # Yoga
            np.random.randint(40, 100, 100)      # Strength
        ]),
        'gender': np.random.choice([0, 1], 300),
        'workout_type': ['Cardio'] * 100 + ['Yoga'] * 100 + ['Strength'] * 100
    })

    le = LabelEncoder()
    data['workout_type_encoded'] = le.fit_transform(data['workout_type'])
    X = data[['age', 'weight', 'height', 'steps_walked', 'workout_duration', 'gender']]
    y = data['workout_type_encoded']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le

model, le = train_model()
workout_type_label = le.classes_

# ---------- Session state ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- Login or Signup Page ----------
st.title("🏋️ Personal Fitness Tracker")

if not st.session_state.logged_in:
    choice = st.radio("Choose an option:", ["Login", "Signup"], horizontal=True)

    if choice == "Login":
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid credentials. Try again.")
        st.stop()

    elif choice == "Signup":
        st.subheader("🆕 Signup")
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        if st.button("Signup"):
            if save_user(new_username, new_password):
                st.success("Signup successful! Please log in.")
            else:
                st.warning("Username already exists. Try a different one.")
        st.stop()

# ---------- Input Section ----------
st.subheader("🧍 Enter Your Fitness Info")
age = st.slider("Age", 18, 60, 25)
weight = st.slider("Weight (kg)", 40, 120, 65)
height = st.slider("Height (cm)", 140, 210, 170)
steps_walked = st.number_input("Steps Walked Today", min_value=0, value=5000)
workout_duration = st.number_input("Workout Duration (minutes)", min_value=0, value=45)
gender = st.radio("Gender", ["Male", "Female"])
gender_val = 0 if gender == "Male" else 1

# ---------- Predict Workout ----------
user_input = pd.DataFrame([[age, weight, height, steps_walked, workout_duration, gender_val]],
                          columns=['age', 'weight', 'height', 'steps_walked', 'workout_duration', 'gender'])

if st.button("🏃 Suggest Workout"):
    prediction = model.predict(user_input)[0]
    predicted_label = workout_type_label[prediction]
    probs = model.predict_proba(user_input)[0]

    st.success(f"✅ Suggested Workout Type: **{predicted_label}**")

    st.subheader("📊 Model Confidence")
    for i, prob in enumerate(probs):
        st.write(f"- {workout_type_label[i]}: {prob * 100:.2f}%")

    # Save prediction to CSV
    history_entry = pd.DataFrame({
        "username": [st.session_state.username],
        "age": [age],
        "weight": [weight],
        "height": [height],
        "steps_walked": [steps_walked],
        "workout_duration": [workout_duration],
        "gender": [gender],
        "prediction": [predicted_label],
        "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    try:
        existing = pd.read_csv("predictions.csv")
        updated = pd.concat([existing, history_entry], ignore_index=True)
    except FileNotFoundError:
        updated = history_entry

    updated.to_csv("predictions.csv", index=False)

# ---------- Show History ----------
st.subheader("📁 Your Prediction History")
try:
    history = pd.read_csv("predictions.csv")
    user_history = history[history['username'] == st.session_state.username]
    st.dataframe(user_history.sort_values("timestamp", ascending=False))
except FileNotFoundError:
    st.info("No prediction history yet.")
