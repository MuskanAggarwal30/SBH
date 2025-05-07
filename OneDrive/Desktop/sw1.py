import streamlit as st
import json
import os
import hashlib
import random
import pandas as pd
#from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
from datetime import timedelta
#import openai
from time import time
from uuid import uuid4
#from flask import Flask, jsonify, request
import folium
from streamlit_folium import st_folium
import cv2
import datetime
import numpy as np
from streamlit_lottie import st_lottie
import requests
st.set_page_config(page_title="Smart Waste Management", layout="centered")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

    lottie_recycle = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_recycling.json")
    st_lottie(lottie_recycle, height=200)

def main_app():
    # Load Lottie Animation
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    # Color Coding for Waste Level
    def color_waste(val):
        if val >= 80:
            color = 'red'
        elif val >= 50:
            color = 'orange'
        else:
            color = 'green'
        return f'background-color: {color}'
    # Simulated Citizen Complaints Database
    complaints_db = []

    # Simulated Admin Announcements Database
    announcements_db = []


    def load_bin_data():
        np.random.seed(42)
        data = {
            'Bin ID': [f'BIN-{i}' for i in range(1, 11)],
            'Latitude': np.random.uniform(12.90, 13.00, 10),
            'Longitude': np.random.uniform(77.50, 77.60, 10),
            'Waste % Full': np.random.randint(30, 100, 10),
            'Area': np.random.choice(['North Zone', 'East Zone', 'South Zone'], 10)
        }
        return pd.DataFrame(data)
    # Load Data
    bin_data = load_bin_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Citizen Portal", "Admin Panel", "Educational Section", "Waste Sorting Quiz"])

    if page == "Dashboard":
        st.title("♻️ Smart Waste Management Dashboard")
        # Lottie Animation
        lottie_url = "https://assets2.lottiefiles.com/packages/lf20_cg3zkg.json"
        lottie = load_lottieurl(lottie_url)
        if lottie:
            st_lottie(lottie, height=150, key="recycle")

        # KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bins", bin_data.shape[0])
        with col2:
            st.metric("Avg Fill Level (%)", f"{bin_data['Waste % Full'].mean():.2f}")
        with col3:
            st.metric("Bins Overflowing", (bin_data['Waste % Full'] > 80).sum())

        st.subheader("📈 Waste Levels Overview")
        styled_df = bin_data.style.applymap(color_waste, subset=['Waste % Full'])
        st.dataframe(styled_df, use_container_width=True)

        # Map
        st.subheader("🗺️ Bin Locations")
        map_center = [bin_data['Latitude'].mean(), bin_data['Longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=13)

        for _, row in bin_data.iterrows():
            color = "green"
            if row['Waste % Full'] >= 80:
                color = "red"
            elif row['Waste % Full'] >= 50:
                color = "orange"
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"Bin ID: {row['Bin ID']} ({row['Waste % Full']}% full)",
                icon=folium.Icon(color=color)
            ).add_to(m)

        st_folium(m, width=700)

        st.title("🚮 Smart Waste Management - ML-Based Waste Prediction")
        # Load or initialize dataset
        @st.cache_data
        def load_initial_data():
            data = {
                'state': ['Delhi', 'Delhi', 'Maharashtra', 'Maharashtra', 'Karnataka'],
                'date': pd.date_range('2024-01-01', periods=5),
                'temperature': [28, 29, 30, 31, 32],
                'population': [2000, 2050, 3000, 3100, 2500],
                'waste_kg': [120, 130, 220, 230, 180]
            }
            return pd.DataFrame(data)
        # Allow user to upload new CSV data
        st.sidebar.header("📤 Upload State-wise Data")
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("New data uploaded!")
        else:
            df = load_initial_data()

        # Convert date column if necessary
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        # Train models per state
        models = {}
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            X = state_df[['temperature', 'population']]
            y = state_df['waste_kg']
            model = LinearRegression()
            model.fit(X, y)
            models[state] = model

        # Sidebar input
        st.sidebar.header("📊 Predict Waste for a State")
        state_list = df['state'].unique()
        selected_state = st.sidebar.selectbox("Choose State", state_list)
        input_temp = st.sidebar.number_input("Temperature (°C)", value=30)
        input_pop = st.sidebar.number_input("Population", value=3000)

        if st.sidebar.button("Predict"):
            model = models[selected_state]
            prediction = model.predict([[input_temp, input_pop]])[0]
            st.sidebar.success(f"Predicted Waste: {prediction:.2f} kg")
        
        # Main section: show data and charts
        st.header(f"📈 Historical Waste Trends for {selected_state}")
        state_df = df[df['state'] == selected_state]
        fig, ax = plt.subplots()
        ax.plot(state_df['date'], state_df['waste_kg'], marker='o', color='green')
        ax.set_title(f"Waste over Time in {selected_state}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Waste (kg)")
        st.pyplot(fig)

        # Optional: show full data table
        with st.expander("🔍 View Data Table"):
            st.dataframe(state_df)
        # Allow download of data
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Current Dataset", csv, "statewise_waste_data.csv", "text/csv")

        st.header(f"📅 7-Day Forecast for {selected_state}")

        # Prepare time series data
        forecast_df = state_df.copy()
        forecast_df = forecast_df.sort_values('date')
        forecast_df['date_ordinal'] = forecast_df['date'].map(pd.Timestamp.toordinal)

        X_time = forecast_df[['date_ordinal']]
        y_time = forecast_df['waste_kg']

        # Fit model
        time_model = LinearRegression()
        time_model.fit(X_time, y_time)

        # Generate future dates
        last_date = forecast_df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_ordinals = [[d.toordinal()] for d in future_dates]

        # Predict
        future_preds = time_model.predict(future_ordinals)

        # Create DataFrame
        forecast_results = pd.DataFrame({
            'date': future_dates,
            'predicted_waste_kg': future_preds
        })

        # Plot
        fig2, ax2 = plt.subplots()
        ax2.plot(forecast_df['date'], forecast_df['waste_kg'], label="Historical", marker='o')
        ax2.plot(forecast_results['date'], forecast_results['predicted_waste_kg'], label="Forecast", marker='x', linestyle='--')
        ax2.set_title(f"7-Day Waste Forecast - {selected_state}")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Waste (kg)")
        ax2.legend()
        st.pyplot(fig2)
        # Optionally show table
        with st.expander("📄 Forecast Data Table"):
            st.dataframe(forecast_results)

        # Optional: File upload to update data
        st.subheader("Upload New Data (CSV Format)")
        uploaded_file = st.file_uploader("Choose a file with Area, Latitude, Longitude, Waste_Level (%)", type=["csv"])
        if uploaded_file:
            new_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(new_data)
 
        st.subheader("📷 Overflow Detection (Upload Bin Image)")
        uploaded_img = st.file_uploader("Upload bin image", type=["jpg", "png", "jpeg"])
        if uploaded_img:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            # Optionally show image
            st.image(image, channels="BGR")
            # Placeholder for future detection logic
            st.info("🔍 (Detection model pending) This will analyze bin status in future.")

    elif page == "Citizen Portal":
        st.title("🙋 Citizen Portal")
        # (Your existing citizen portal code here...)
        st.write("Submit Complaints, Suggestions, and More!")

        with st.form("complaint_form"):
            name = st.text_input("Name")
            area = st.selectbox("Area", bin_data['Area'].unique())
            complaint = st.text_area("Complaint / Suggestion")
            submit = st.form_submit_button("Submit")
            if submit:
                complaints_db.append({'Name': name, 'Area': area, 'Complaint': complaint})
                st.success("✅ Complaint submitted successfully!")

        st.divider()
        st.subheader("📝 View Submitted Complaints (For Reference)")
        if complaints_db:
            st.table(pd.DataFrame(complaints_db))
        else:
            st.info("No complaints submitted yet.")

    elif page == "Admin Panel":
        st.title("🛠️ Admin Panel")
        # (Your existing admin panel code here...)
        st.write("Manage Announcements, View Complaints, and More!")
        admin_password = st.text_input("Enter Admin Password", type="password")
        if admin_password == "admin123":
            st.success("🔓 Access Granted")
                # View complaints
            st.subheader("📋 Citizen Complaints")
            if complaints_db:
                st.dataframe(pd.DataFrame(complaints_db), use_container_width=True)
            else:
                st.info("No complaints yet.")

            # Admin Announcement
            st.subheader("📢 Post Announcement to Citizens")
            with st.form("announcement_form"):
                announcement = st.text_area("Enter announcement text...")
                post = st.form_submit_button("Post")

                if post:
                    announcements_db.append(announcement)
                    st.success("✅ Announcement posted!")

            st.subheader("📋 All Announcements")
            if announcements_db:
                for i, ann in enumerate(announcements_db, 1):
                    st.info(f"({i}) {ann}")
            else:
                    st.info("No announcements posted yet.")

        else:
            st.warning("🔒 Admin access restricted.")

    elif page == "Educational Section":
        #educational_section()  # Display educational content
        st.subheader("🎥 Watch This Video to Learn About Waste Sorting")
        video_url = "https://www.youtube.com/embed/dQw4w9WgXcQ"  # Sample YouTube video link
        st.markdown(f'<iframe width="700" height="400" src="{video_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
    
        # Infographic
        st.subheader("📊 Waste Sorting Infographic")
        infographic_url = "https://example.com/infographic_image.jpg"  # Link to an infographic
        st.image(infographic_url, caption="Infographic on Waste Sorting", use_column_width=True)
    
        # Educational Article
        st.subheader("📝 Read This Article on Waste Management")
        article_content = """
        Waste management is crucial to keeping our environment clean. Sorting waste helps to recycle and reuse materials effectively. 
        The key steps to sorting waste include:
    
        - **Plastic**: Recycle plastic bottles, containers, and packaging.
        - **Glass**: Glass bottles and jars can be recycled infinitely.
        - **Paper**: Recycle newspapers, magazines, and cardboard.
        - **Organic Waste**: Compost food scraps to reduce landfill waste.
    
        Learn more about how you can reduce, reuse, and recycle to keep our planet clean!
        """
        st.markdown(article_content)

    elif page == "Waste Sorting Quiz":
        #waste_sorting_quiz()  # Start the waste sorting quiz
        questions = [
            {
                "question": "Which of these items can be recycled?",
                "options": ["Plastic Bottles", "Banana Peel", "Pizza Box", "Used Tissue"],
                "answer": "Plastic Bottles"
            },
            {
                "question": "Where should food scraps go?",
                "options": ["Plastic Bin", "Organic Waste Bin", "General Waste Bin", "Glass Bin"],
                "answer": "Organic Waste Bin"
            },
            {
                "question": "Which material is best for recycling?",
                "options": ["Plastic Bags", "Aluminum Cans", "Food Waste", "Styrofoam"],
                "answer": "Aluminum Cans"
            },
            {
                "question": "Which of these cannot be recycled?",
                "options": ["Glass Bottle", "Plastic Container", "Used Tissue", "Newspaper"],
                "answer": "Used Tissue"
            }
        ]
        total_questions = len(questions)

        # --- Initialize session state ---
        if "current_question" not in st.session_state:
            st.session_state.current_question = 0
        if "user_answers" not in st.session_state:
            st.session_state.user_answers = [None] * total_questions
        if "quiz_completed" not in st.session_state:
            st.session_state.quiz_completed = False

        # --- Quiz Interface ---
        if not st.session_state.quiz_completed:
            q = questions[st.session_state.current_question]
            st.subheader(f"Q{st.session_state.current_question + 1}: {q['question']}")
            selected_option = st.radio("Choose an option:", q["options"], key=f"q{st.session_state.current_question}")

            # Save selected answer
            st.session_state.user_answers[st.session_state.current_question] = selected_option

            # Next button
            if st.button("Next"):
                if st.session_state.current_question < total_questions - 1:
                    st.session_state.current_question += 1
                else:
                    st.warning("You're on the last question. Please Submit.")

            # Submit button (only visible on last question)
            if st.session_state.current_question == total_questions - 1:
                if st.button("Submit"):
                    st.session_state.quiz_completed = True

        else:
             # --- Results after submit ---
             score = 0
             for idx, user_ans in enumerate(st.session_state.user_answers):
                 if user_ans == questions[idx]["answer"]:
                     score += 1

             st.success(f"🎯 Your Score: {score} out of {total_questions}")
             if score == total_questions:
                 st.balloons()
                 st.success("🏆 Perfect! You're a Waste Sorting Champion!")
             elif score >= total_questions // 2:
                 st.warning("👏 Good job! But you can learn even more!")
             else:
                 st.error("📚 Keep learning! Try again to improve your knowledge.")
             if st.button("Restart Quiz"):
                 st.session_state.current_question = 0
                 st.session_state.user_answers = [None] * total_questions
                 st.session_state.quiz_completed = False
    


# -------------------------------
# 🔐 AUTHENTICATION & USER MGMT
# -------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def register_user(username, password, role="user"):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": hash_password(password),
        "role": role
    }
    save_users(users)
    return True

def login_user(username, password):
    users = load_users()
    user = users.get(username)
    if not user:
        return False, None
    if user["password"] == hash_password(password):
        return True, user["role"]
    return False, None

# -------------------------------
# 📊 WASTE DATA SIMULATION
# -------------------------------
def get_simulated_data():
    areas = ['Area A', 'Area B', 'Area C', 'Area D']
    data = {
        'Area': areas,
        'Waste Level (%)': [random.randint(30, 100) for _ in areas],
        'Predicted Days to Full': [random.randint(1, 7) for _ in areas]
    }
    return pd.DataFrame(data)

# -------------------------------
# 📋 ADMIN PANEL
# -------------------------------
def show_admin_panel():
    st.subheader("🛠 Admin Panel")

    users = load_users()
    st.markdown("### 👥 Registered Users")
    st.table(pd.DataFrame([
        {"Username": u, "Role": users[u]["role"]}
        for u in users
    ]))

    st.markdown("---")
    st.markdown("### ♻️ Simulated Bin Management")
    bins = get_simulated_data()
    st.dataframe(bins)

    for i in range(len(bins)):
        print(f"Creating button with key: empty_bins_button_{i}")

# -------------------------------
# 📊 Admin Analytics Panel
# -------------------------------
def show_admin_analytics():
    st.subheader("📈 Admin Analytics Dashboard")

    # Bin Waste Analysis
    st.markdown("### ♻️ Bin Waste Overview")
    bin_data = get_simulated_data()
    st.write("**Average Waste Level:**", f"{bin_data['Waste Level (%)'].mean():.2f}%")
    st.write("**Maximum Waste Level:**", f"{bin_data['Waste Level (%)'].max()}% in {bin_data.loc[bin_data['Waste Level (%)'].idxmax()]['Area']}")
    
    # Pie Chart of Waste Fill Levels
    bins_low = bin_data[bin_data['Waste Level (%)'] < 50].shape[0]
    bins_medium = bin_data[(bin_data['Waste Level (%)'] >= 50) & (bin_data['Waste Level (%)'] < 75)].shape[0]
    bins_high = bin_data[bin_data['Waste Level (%)'] >= 75].shape[0]
    
    fig, ax = plt.subplots()
    ax.pie([bins_low, bins_medium, bins_high],
           labels=['Low (<50%)', 'Medium (50-75%)', 'High (75%+)'],
           colors=['green', 'orange', 'red'],
           autopct='%1.1f%%', startangle=90)
    ax.set_title("Bin Fill Status")
    st.pyplot(fig)

    # Recycling Leaderboard Insights
    st.markdown("### 🌍 Recycling Stats Overview")
    leaderboard_df = load_data()
    total_participants = leaderboard_df.shape[0]
    total_points = leaderboard_df["Points"].sum()
    top_city = leaderboard_df.groupby("City")["Points"].sum().idxmax()
    
    st.write(f"**Total Participants:** {total_participants}")
    st.write(f"**Total Recycling Points Earned:** {total_points}")
    st.write(f"**Top Performing City:** {top_city}")

    # Time Trend of Recycling
    leaderboard_df['Last Recycle'] = pd.to_datetime(leaderboard_df['Last Recycle'], format="%d-%m-%Y")
    trend = leaderboard_df.groupby("Last Recycle")["Points"].sum().reset_index()

    fig2, ax2 = plt.subplots()
    ax2.plot(trend["Last Recycle"], trend["Points"], marker='o', color='purple')
    ax2.set_title("Recycling Activity Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Points")
    st.pyplot(fig2)


#------------------------------------------------
#   state-wise data
#------------------------------------------------
def state_data():
    st.title("🚮 Smart Waste Management - ML-Based Waste Prediction")

    # Load or initialize dataset
    @st.cache_data
    def load_initial_data():
        data = {
            'state': ['Delhi', 'Delhi', 'Maharashtra', 'Maharashtra', 'Karnataka'],
            'date': pd.date_range('2024-01-01', periods=5),
            'temperature': [28, 29, 30, 31, 32],
            'population': [2000, 2050, 3000, 3100, 2500],
            'waste_kg': [120, 130, 220, 230, 180]
            }
        return pd.DataFrame(data)
    # Allow user to upload new CSV data
    st.sidebar.header("📤 Upload State-wise Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("New data uploaded!")
    else:
        df = load_initial_data()

    # Convert date column if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    # Train models per state
    models = {}
    for state in df['state'].unique():
        state_df = df[df['state'] == state]
        X = state_df[['temperature', 'population']]
        y = state_df['waste_kg']
        model = LinearRegression()
        model.fit(X, y)
        models[state] = model

    # Sidebar input
    st.sidebar.header("📊 Predict Waste for a State")
    state_list = df['state'].unique()
    selected_state = st.sidebar.selectbox("Choose State", state_list)
    input_temp = st.sidebar.number_input("Temperature (°C)", value=30)
    input_pop = st.sidebar.number_input("Population", value=3000)

    if st.sidebar.button("Predict"):
        model = models[selected_state]
        prediction = model.predict([[input_temp, input_pop]])[0]
        st.sidebar.success(f"Predicted Waste: {prediction:.2f} kg")

    
# -------------------------------
# 📈 USER DASHBOARD
# -------------------------------
def show_user_dashboard(username):
    st.subheader(f"📊 Waste Dashboard for {username}")
    df = get_simulated_data()
    st.dataframe(df)
    #dg= state_data()
    #st.dataframe(dg)
    dh=main_app()
    st.dataframe(dh)
    st.markdown("### 📊 Waste Level Chart")
    st.bar_chart(df.set_index("Area")["Waste Level (%)"])

    st.markdown("### 🚛 Suggested Pickup Route")
    route = df.sort_values(by='Waste Level (%)', ascending=False)["Area"].tolist()
    for i, area in enumerate(route, 1):
        st.write(f"{i}. {area}")

# Simulated or uploaded recycling data
@st.cache_data
def load_data():
    return pd.DataFrame({
        "Name": ["Aarav", "Priya", "Rohan", "Isha", "Anaya"],
        "City": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"],
        "Points": [120, 150, 90, 130, 110],
        "Last Recycle": [
            "20-04-2025", "19-04-2025", "21-04-2025", "18-04-2025", "20-04-2025"
        ]
    })

df = load_data()



# -------------------------------------------------------------------------------------
# 🚀 STREAMLIT APP
# -------------------------------------------------------------------------------------

st.title("♻️ Smart Waste Management System")

menu = st.sidebar.selectbox("Menu", ["Login", "Register"])
session = st.session_state

if "logged_in" not in session:
    session.logged_in = False
    session.username = ""
    session.role = ""

if menu == "Register":
    st.subheader("📝 Register")
    new_user = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    role = st.selectbox("Select Role", ["user", "admin"])
    if st.button("Register"):
        if register_user(new_user, new_password, role):
            st.success("Account created successfully!")
        else:
            st.error("Username already exists.")

elif menu == "Login":
    if not session.logged_in:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, role = login_user(username, password)
            if success:
                session.logged_in = True
                session.username = username
                session.role = role
                st.success(f"Welcome, {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")
    else:
        st.sidebar.success(f"Logged in as: {session.username} ({session.role})")
        if session.role == "admin":
            show_admin_panel()
        else:
            show_user_dashboard(session.username)
        if st.button("Logout"):
            session.logged_in = False
            session.username = ""
            session.role = ""
            st.experimental_rerun()

if session.role == "admin":
    show_admin_analytics()
