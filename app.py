import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import bcrypt
from PIL import Image
import os
import io
from datetime import datetime
import matplotlib.pyplot as plt  # For charts
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle  # For PDF generation
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader  # To embed chart in PDF
from reportlab.platypus import Image
import smtplib
from email.message import EmailMessage


# ========== Constants ==========
LOGO_PATH = 'logo.png'
MODEL_PATH = 'heart_disease.pkl'
DATABASE_NAME = 'users.db'

# ========== Database Setup ==========
def init_database():
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT
            )
        ''')
        # ✅ Create predictions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                patient_name TEXT,
                doctor_name TEXT,
                risk_score REAL,
                prediction_result TEXT,
                timestamp TEXT
            )
        ''')
       
        conn.commit()
        
init_database()
# ========== Helper Functions ==========
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def add_user(username, password):
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        password_hash = hash_password(password)
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                 (username, password_hash))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        return c.fetchone()
def send_email_report(recipient_email, pdf_buffer, patient_name):
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"

    msg = EmailMessage()
    msg['Subject'] = 'Heart Disease Prediction Report'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(f"Hello,\n\nAttached is the heart disease prediction report for {patient_name}.\n\nBest regards.")

    pdf_data = pdf_buffer.getvalue()
    msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename=f"{patient_name.replace(' ', '_')}_report.pdf")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def save_prediction(username, patient_name, doctor_name, risk_score, result):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DATABASE_NAME) as conn:
         c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (username, patient_name, doctor_name, risk_score, prediction_result, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, patient_name, doctor_name, risk_score, result, timestamp))
        conn.commit()

def show_live_dashboard():
    st.title("📊 Fresh Hearts Dashboard")

    # Load data
    with sqlite3.connect(DATABASE_NAME) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)

    if df.empty:
        st.info("No predictions recorded yet.")
        return

    # Convert timestamp column
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Sidebar Filters ---
    st.sidebar.header("📁 Filter Predictions")
    doctor_filter = st.sidebar.selectbox("Doctor", options=["All"] + sorted(df['doctor_name'].unique().tolist()))
    risk_filter = st.sidebar.selectbox("Risk Level", options=["All", "High Risk", "Low Risk"])
    date_range = st.sidebar.date_input("Date Range", value=[df['timestamp'].min().date(), df['timestamp'].max().date()])

    # Apply filters
    filtered_df = df.copy()
    if doctor_filter != "All":
        filtered_df = filtered_df[filtered_df['doctor_name'] == doctor_filter]
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['prediction_result'] == risk_filter]
    if date_range:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date.date()) & (filtered_df['timestamp'].dt.date <= end_date.date())]

    if filtered_df.empty:
        st.warning("No results match the selected filters.")
        return

    # --- Summary Metrics ---
    total = len(filtered_df)
    high_risk = len(filtered_df[filtered_df['prediction_result'] == 'High Risk'])
    low_risk = len(filtered_df[filtered_df['prediction_result'] == 'Low Risk'])
    avg_risk = filtered_df['risk_score'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total)
    col2.metric("High Risk Cases", high_risk)
    col3.metric("Avg Risk Score", f"{avg_risk:.2f}%")

    # --- Risk Trend ---
    st.subheader("📈 Risk Trend Over Time")
    df_chart = filtered_df[['timestamp', 'risk_score']].set_index('timestamp').sort_index()
    st.line_chart(df_chart)

    # --- Prediction Log ---
    st.subheader("📋 Filtered Prediction Log")
    st.dataframe(filtered_df, use_container_width=True)

    # --- CSV Export ---
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_predictions.csv',
        mime='text/csv'
    )

    # --- Chart Type Selector ---
    chart_type = st.selectbox("📊 Choose chart type to display & include in PDF:", ["Pie Chart", "Bar Chart"])

    # --- Chart Display ---
    st.subheader("🧮 Prediction Distribution Chart")
    if 'prediction_result' in filtered_df.columns:
        pred_counts = filtered_df['prediction_result'].value_counts()
        labels = pred_counts.index.tolist()
        values = pred_counts.values

        fig, ax = plt.subplots(figsize=(5, 4))

        if chart_type == "Pie Chart":
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            plt.title('Prediction Distribution (Pie)')
        elif chart_type == "Bar Chart":
            colors_map = ['green' if lbl == 'Low Risk' else 'red' for lbl in labels]
            ax.bar(labels, values, color=colors_map)
            plt.title('Prediction Distribution (Bar)')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

        st.pyplot(fig)

        # Save chart to image buffer for PDF
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='PNG')
        plt.close()
        img_buf.seek(0)
    else:
        st.info("No prediction data available for chart.")
        img_buf = None

    # --- PDF Export ---
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    img = Image("logo.png", width=100, height=50)  # ✅ Use correct Flowable
    elements.append(img)

    elements.append(Paragraph("📊 Fresh Hearts - Prediction Log Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Table section
    if not filtered_df.empty:
        data = [filtered_df.columns.tolist()] + filtered_df.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No data available for the selected filters.", styles['Normal']))

    # Chart section
    if img_buf:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("🧮 " + chart_type, styles['Heading2']))
        elements.append(Spacer(1, 6))
        chart = Image(img_buf,  width=400, height=300)
        elements.append(chart)

    # Build PDF and make downloadable
    doc.build(elements)
    pdf_buffer.seek(0)

    st.download_button(
        label="🖨️ Download Dashboard Summary (PDF)",
        data=pdf_buffer,
        file_name="dashboard_summary.pdf",
        mime="application/pdf"
    )

# ========== Authentication ==========
def show_auth():
    st.markdown("## 🚀 Get Started with Fresh Hearts")
    st.write("Enter details below to begin your journey towards a healthier tomorrow.")

    menu = st.sidebar.radio("Menu", ["Login", "Sign Up"])

    if menu == "Login":
        with st.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.form_submit_button("Login"):
                user = get_user(username)
                if user and verify_password(password, user[1]):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    elif menu == "Sign Up":
        with st.form("Sign Up Form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type='password')
            if st.form_submit_button("Create Account"):
                if get_user(new_user):
                    st.warning("Username exists!")
                else:
                    add_user(new_user, new_pass)
                    st.success("Account created! Please login.")

# ========== Main Application ==========
def load_model():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    # Patch missing attribute for scikit-learn compatibility
    for estimator in model.estimators_:
        if not hasattr(estimator, 'monotonic_cst'):
            estimator.monotonic_cst = None
    return model

# show_main_app()
def show_main_app():
    model = load_model()

    st.image(LOGO_PATH, width=200)
    st.markdown("<h1 style='text-align: center; color: #e63946;'>Fresh Hearts - Heart Disease Predictor</h1>", 
                unsafe_allow_html=True)

    nav = st.sidebar.radio("Navigation", ["Predict", "Live Dashboard"])

    if nav == "Predict":
        st.write("This app predicts your risk of heart disease based on clinical parameters.")
        with st.sidebar:
            st.header('Patient Information')
            patient_name = st.text_input("Patient Name")
            doctor_name = st.text_input("Doctor Name")
            input_data = get_user_inputs()
        st.subheader('Patient Information Summary')
        st.dataframe(pd.DataFrame(input_data, index=[0]), use_container_width=True)
        if st.button('Predict Heart Disease Risk'):
            if not patient_name or not doctor_name:
                st.error("Please enter both the Patient Name and Doctor Name before predicting.")
            else:
                make_prediction(model, input_data, patient_name, doctor_name)
    elif nav == "Live Dashboard":
        show_live_dashboard()

# ========== Input Components ==========

def get_user_inputs(prefix="input"):
    return {
        'age': st.sidebar.number_input('Age', 1, 120, 50, key=f"{prefix}_age"),
        'sex': st.sidebar.selectbox('Sex', options=[0, 1], key=f"{prefix}_sex"),
        'cp': st.sidebar.selectbox('Chest Pain Type', options=[0, 1, 2, 3], key=f"{prefix}_cp"),
        'trestbps': st.sidebar.number_input('Resting BP (mm Hg)', 80, 200, 120, key=f"{prefix}_trestbps"),
        'chol': st.sidebar.number_input('Cholesterol (mg/dl)', 100, 600, 200, key=f"{prefix}_chol"),
        'fbs': st.sidebar.selectbox('Fasting Blood Sugar >120', options=[0, 1], key=f"{prefix}_fbs"),
        'restecg': st.sidebar.selectbox('ECG Results', options=[0, 1, 2], key=f"{prefix}_restecg"),
        'thalach': st.sidebar.number_input('Max Heart Rate', 60, 220, 150, key=f"{prefix}_thalach"),
        'exang': st.sidebar.selectbox('Exercise Angina', options=[0, 1], key=f"{prefix}_exang"),
        'oldpeak': st.sidebar.number_input('ST Depression', 0.0, 10.0, 1.0, 0.1, key=f"{prefix}_oldpeak"),
        'slope': st.sidebar.selectbox('ST Slope', options=[0, 1, 2], key=f"{prefix}_slope"),
        'ca': st.sidebar.selectbox('Major Vessels', [0, 1, 2, 3], key=f"{prefix}_ca"),
        'thal': st.sidebar.selectbox('Thalassemia', options=[1, 2, 3], key=f"{prefix}_thal")
    }

def add_footer(c, width):
    """Function to add footer text at bottom of page."""
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    footer_text = "Confidential - For Authorized Medical Use Only"
    c.drawCentredString(width / 2, 20, footer_text)

# ========== Prediction Logic ==========
def make_prediction(model, input_data, patient_name, doctor_name):
    input_df = pd.DataFrame({k: [v] for k, v in input_data.items()})

    # Make the prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.session_state['total_predictions'] = st.session_state.get('total_predictions', 0) + 1
    if prediction[0] == 1:
        st.session_state['high_risk_count'] = st.session_state.get('high_risk_count', 0) + 1

    if prediction[0] == 1:
        result_text = "🚨 Heart Disease Detected!"
    else:
        result_text = "✅ No Heart Disease."

    risk_percentage = probability[0][1] * 100
    prev_total = st.session_state['total_predictions']
    prev_avg = st.session_state.get('average_risk', 0)
    st.session_state['average_risk'] = ((prev_avg * (prev_total - 1)) + risk_percentage) / prev_total

    st.metric("Heart Disease Risk Probability", f"{risk_percentage:.2f}%")
    st.subheader("Prediction Result")
    st.write(f"Prediction: **{result_text}**")

    save_prediction(
        username=st.session_state.get('username', 'anonymous'),
        patient_name=patient_name,
        doctor_name=doctor_name,
        risk_score=risk_percentage,
        result="High Risk" if prediction[0] == 1 else "Low Risk"
    )
    
    # --- PDF Generation ---
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # (Optional) Logo
    if os.path.exists(LOGO_PATH):
     c.drawImage(LOGO_PATH, 50, height - 100, width=100, preserveAspectRatio=True)
    else:
     st.warning("Logo not found. Skipping logo in the PDF report.")
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.darkblue)
    c.drawString(width / 4, height - 50, "Heart Disease Prediction Report")

    # Subheading
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(50, height - 140, "Fresh Hearts Diagnosis Report")
   
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 160, f"Patient Name: {patient_name}")
    c.drawString(300, height - 160, f"Doctor Name: {doctor_name}")
    c.drawString(50, height - 180, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.red if prediction[0] == 1 else colors.green)
    c.drawString(50, height - 210, f"Prediction: {result_text}")

    # Risk
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawString(50, height - 230, f"Heart Disease Risk: {risk_percentage:.2f}%")

    # Input Data Section
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 260, "Input Data:")

    # Draw table headers
    c.setFont("Helvetica-Bold", 11)
    x_left = 50
    x_right = 300
    y = height - 280
    row_height = 18
    c.drawString(x_left, y, "Parameter")
    c.drawString(x_right, y, "Value")
    y -= row_height
    c.line(x_left, y + 10, x_right + 300, y + 10)  # underline headers
# Draw data rows
    c.setFont("Helvetica", 11)
    for key, value in input_data.items():
     c.drawString(x_left, y, str(key).capitalize())
     c.drawString(x_right, y, str(value))
     y -= row_height

    if y < 80:
        add_footer(c, width)
        c.showPage()
        y = height - 50
        c.setFont("Helvetica", 11)

    add_footer(c, width)  # Add footer on the last page
    c.save()
    pdf_buffer.seek(0)

    # --- Download button for PDF ---
    st.download_button(
        label="📥 Download Full Report (PDF)",
        data=pdf_buffer,
        file_name=f"{patient_name.replace(' ', '_')}_heart_report.pdf",
        mime="application/pdf"
    )

    recipient_email = st.text_input("Enter recipient email", key="email_input")
    if recipient_email:
        if send_email_report(recipient_email, pdf_buffer, patient_name):
            st.success(f"📧 Report sent successfully to {recipient_email}")

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('⚠️ High Risk of Heart Disease!')
    else:
        st.success('✅ Low Risk of Heart Disease!')

    # Probability Visualization
    st.subheader('Risk Probability')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Heart Disease", f"{probability[0][1]*100:.2f}%")
    with col2:
        st.metric("No Heart Disease", f"{probability[0][0]*100:.2f}%")
    
    # Medical Recommendations
    st.subheader('Recommendations')
    if prediction[0] == 1:
        st.warning(high_risk_recommendations())
    else:
        st.info(low_risk_recommendations())

def high_risk_recommendations():
    return """
    - Consult a cardiologist immediately
    - Adopt heart-healthy diet (low salt/cholesterol)
    - Engage in approved physical activity
    - Monitor vitals regularly
    - Avoid smoking/alcohol
    """

def low_risk_recommendations():
    return """
    - Maintain healthy lifestyle
    - Regular checkups recommended
    - Manage stress levels
    - Continue preventive care
    """

# ========== Main Execution ==========
if __name__ == "__main__":
    # Initial Setup

# Page Configuration
 st.set_page_config(
    page_title="Fresh Hearts",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === Custom CSS Styling ===
st.markdown("""
    <style>
    /* Fullscreen animated gradient background */
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(270deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c1ee, #fddb92);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Segoe UI', sans-serif;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main .block-container {
        padding-top: 2rem;
        background-color: rgba(255, 255, 255, 0.8);  /* Light transparent card feel */
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }

    .custom-header {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #D00000;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeIn 2s ease-in;
        margin-bottom: 20px;
    }

    .login-header {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        color: #222;
        text-shadow: 1px 1px 3px #ffe6e6;
        animation: fadeIn 1.5s ease-in;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        font-size: 0.85em;
        color: gray;
        margin-top: 3em;
        padding-bottom: 2em;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)
# === Session State ===
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# === Login Flow ===
if not st.session_state.logged_in:
    # 🎯 Custom Header for Login Page
    st.markdown('<div class="login-header">Heart Disease Prediction AI System </div>', unsafe_allow_html=True)
    show_auth()  # Your login form
else:
    # 🎯 Logout Button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    # 🎉 Welcome Only on Main Page
    st.markdown('<div class="custom-header">Welcome to ❤️ Fresh Hearts AI System</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:2.2em; font-style:italic; color:#444;">🌿 A healthy tomorrow matters</div>', unsafe_allow_html=True)
    st.markdown("##")
    show_main_app()  # Main logic

# === Footer ===
st.markdown('<div class="footer">Developed with ❤️ by <strong>SKD Fresh Hearts Team</strong></div>', unsafe_allow_html=True)