# Personal Fitness Tracker Using Python

This is a Streamlit web application that suggests workout types based on input parameters like age, height, weight, gender, and BMI.

## 🚀 Features

- User Signup & Login
- Predict workout suggestion (Yoga, Cardio, Strength)
- Save predictions per user
- View login and prediction logs

## 🛠️ Installation

1. **Clone the repository**  
```bash
git clone https://github.com/Sandeep-Udbal/Personal-Fitness-Tracker.git 
cd fitness-dashboard
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Run the application**  
```bash
streamlit run app.py
```

## 🗃️ Files Created

- `users.csv`: Stores user credentials (username, password)
- `predictions.csv`: Logs user predictions with input data

These files are automatically generated in the root directory when you run the app.

## 📁 Folder Structure

```
fitness-dashboard/
├── app.py
├── users.csv (auto-created)
├── predictions.csv (auto-created)
└── requirements.txt
```

## 📋 Requirements

- Python 3.7 or above
- Streamlit
- Pandas

## 📌 Note

- All user data is stored locally using CSV files.
- This is a demo app and does not use encrypted passwords — avoid using real credentials.

## 📬 Contact

For issues or suggestions, raise a GitHub issue or contact me.

---

Made with ❤️ using Streamlit.