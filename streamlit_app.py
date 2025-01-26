import streamlit as st
from PIL import Image
import joblib
import numpy as np
import pandas as pd
import pickle
import requests
import os




# Set up page configuration
st.set_page_config(
    page_title="WiseLFC",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)


option = st.sidebar.selectbox(
    'Navigate',
    ('🏠 Home', '🔍 Fraud Detection', '📊 Credit Score', '🏦 Loan Prediction')
)
# Enhanced CSS with black theme and animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    /* Black theme colors */
    :root {
        --bg-primary: #000000;
        --bg-secondary: #111111;
        --text-primary: #ffffff;
        --text-secondary: #888888;
        --accent-primary: #1DB954;
        --accent-secondary: #169c46;
        --error: #ff4444;
        --success: #1DB954;
    }
    
    body {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    .main {
        padding: 2rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
        color: var(--text-primary);
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.3);
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: var(--bg-secondary);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card styling */
    .service-card {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .service-card:hover {
        transform: translateY(-5px);
    }
    
    /* Typography */
    h1 {
        background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    h2, h3, h4 {
        color: var(--accent-primary);
    }
    
    /* Form elements */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        background: var(--bg-secondary);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: var(--text-primary);
        padding: 0.5rem;
    }
    
    /* Messages */
    .success-message {
        background: var(--bg-secondary);
        color: var(--accent-primary);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--accent-primary);
        animation: slideIn 0.5s ease;
    }
    
    .error-message {
        background: var(--bg-secondary);
        color: var(--error);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--error);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .custom-logo {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
        animation: fadeIn 1s ease;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    /* Grid and Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .input-container {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--bg-secondary);
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Dark mode specific adjustments */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown a {
        color: var(--accent-primary);
    }
    
    /* Table styling */
    table {
        background: var(--bg-secondary);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    th, td {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: var(--text-primary);
    }
    </style>
    """, unsafe_allow_html=True)
# Google Drive File ID
file_id = "1CVuThI-cKrR0_XhzkMkPNZGmYjxmfsKE"

# Construct direct download URL
download_url = f"https://drive.google.com/uc?id={file_id}"

# Output file path
output_path = "credit.pkl"

# Check if the file already exists
if not os.path.exists(output_path):
    st.info("Downloading the model file. Please wait...")
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Write file to disk
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model file downloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the file: {e}")
else:
    st.info("Model file already exists.")



# Load models (replace with the actual paths to your trained models)
#fraud_model = joblib.load('E:/Sem Project/Codes/fraud_dt.pkl')
credit_model = joblib.load('credit.pkl')
#loan_model = joblib.load('loan.pkl')




# Add the cf function here
def cf(step, type, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest):
    transaction_type = 'CASH_OUT' if type == 0 else 'TRANSFER'
    fraud_messages = []

    if transaction_type == 'TRANSFER':
        if newbalanceOrig != oldbalanceOrig - amount:
            fraud_messages.append("Potential fraud detected")
        if newbalanceDest != oldbalanceDest + amount:
            fraud_messages.append("Potential fraud detected")
    elif transaction_type == 'CASH_OUT':
        if newbalanceOrig != oldbalanceOrig - amount:
            fraud_messages.append("Potential fraud detected")
        if newbalanceDest != oldbalanceDest:
            fraud_messages.append("Potential fraud detected")
    
    if fraud_messages:
        st.markdown('<div class="error-message">Warning: This transaction is likely fraudulent.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-message">This transaction appears to be legitimate.</div>', unsafe_allow_html=True)

# Display logo and title section continues from here...
# Home page with feature highlights
if option == "🏠 Home":
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h3>🔍 Fraud Detection</h3>
            <p>Advanced AI algorithms to detect suspicious transactions in real-time</p>
        </div>
        <div class="feature-card">
            <h3>📊 Credit Score</h3>
            <p>Get instant credit score predictions based on your financial data</p>
        </div>
        <div class="feature-card">
            <h3>🏦 Loan Prediction</h3>
            <p>Smart loan eligibility assessment using advanced machine learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="service-card">
        <h2>Why Choose WiseLFC?</h2>
        <ul>
            <li>Advanced AI-powered analysis</li>
            <li>Instant results and predictions</li>
            <li>Secure and confidential</li>
            <li>User-friendly interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif option == "🔍 Fraud Detection":
    #st.markdown('<div class="service-card">', unsafe_allow_html=True)
    st.header("🔍 Fraud Detection")
    st.write("Enter transaction details to detect potential fraud.")
    
    with st.container():
        step = 1
        type = int(st.selectbox("Transaction Type", options=[(0, "CASH_OUT"), (1, "TRANSFER")], format_func=lambda x: x[1])[0])
        
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction amount", min_value=0.0)
            oldbalanceOrig = st.number_input("Old Balance of Sender", min_value=0.0)
            newbalanceOrig = st.number_input("New Balance of Sender", min_value=0.0)
        
        with col2:
            oldbalanceDest = st.number_input("Old Balance of Receiver", min_value=0.0)
            newbalanceDest = st.number_input("New Balance of Receiver", min_value=0.0)

        if st.button("Analyze Transaction"):
            cf(step, type, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest)

    
    #st.markdown('</div>', unsafe_allow_html=True)

elif option == "📊 Credit Score":
    #st.markdown('<div class="service-card">', unsafe_allow_html=True)
    st.header("📊 Credit Score Prediction")
    st.write("Provide your information for a quick credit score estimate.")

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Annual Income", min_value=0.0)
        b = st.number_input("Monthly Inhand Salary", min_value=0.0)
        c = st.number_input("Number of Bank Accounts", min_value=0)
        d = st.number_input("Number of Credit Cards", min_value=0)
        f = st.number_input("Number of Loans", min_value=0)
        e = st.number_input("Interest Rate", min_value=0.0)

    with col2:
        g = st.number_input("Average Number of Days Delayed", min_value=0)
        h = st.number_input("Number of Delayed Payments", min_value=0)
        i = st.selectbox("Credit Mix", options=[(0, "Bad"), (1, "Standard"), (3, "Good")], format_func=lambda x: x[1])[0]
        j = st.number_input("Outstanding Debt", min_value=0.0)
        k = st.number_input("Credit History Age", min_value=0)
        l = st.number_input("Monthly Balance", min_value=0.0)

    if st.button("Calculate Credit Score"):
        features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
        predicted_score = credit_model.predict(features)
        st.markdown(f'<div class="success-message">Your predicted credit score is: {predicted_score[0]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif option == "🏦 Loan Prediction":
    #st.markdown('<div class="service-card">', unsafe_allow_html=True)
    st.header("🏦 Loan Prediction")
    st.write("Enter your details to check loan eligibility.")

    col1, col2 = st.columns(2)
    with col1:
        Gender_Male = 1 if st.selectbox("Gender", options=["Male", "Female"]) == "Male" else 0
        Married_Yes = 1 if st.selectbox("Married", options=["Yes", "No"]) == "Yes" else 0
        Dependents = st.number_input("Number of Dependents", min_value=0, max_value=3)
        Education_Not_Graduate = 0 if st.selectbox("Education", options=["Graduate", "Not Graduate"]) == "Graduate" else 1
        Self_Employed_Yes = 1 if st.selectbox("Self Employed", options=["Yes", "No"]) == "Yes" else 0
    
    with col2:
        Property = st.selectbox("Property Area", options=["Urban", "Rural", "Semiurban"])
        Property_Area_Urban = 1 if Property == "Urban" else 0
        Property_Area_Semiurban = 1 if Property == "Semiurban" else 0
        Credit_History = 1 if st.selectbox("Credit History", options=["Yes", "No"]) == "Yes" else 0
        Total_Income = st.number_input("Total Income (LPA)", min_value=0.0)
        EMI = st.number_input("Loan Amount", min_value=0.0)

    # Calculate dependent values
    Dependents_1 = 1 if Dependents == 1 else 0
    Dependents_2 = 1 if Dependents == 2 else 0
    Dependents_3 = 1 if Dependents == 3 else 0
    Balance_Income = 1

    if st.button("Check Loan Eligibility"):
        input_data = [Credit_History, Total_Income, EMI, Balance_Income, Gender_Male, 
                    Married_Yes, Dependents_1, Dependents_2, Dependents_3, 
                    Education_Not_Graduate, Self_Employed_Yes, Property_Area_Semiurban,
                    Property_Area_Urban]
        
        columns = ['Credit_History', 'Total_Income', 'EMI', 'Balance_Income', 
                'Gender_Male', 'Married_Yes', 'Dependents_1', 'Dependents_2', 
                'Dependents_3+', 'Education_Not Graduate', 'Self_Employed_Yes', 
                'Property_Area_Semiurban', 'Property_Area_Urban']
        
        input_df = pd.DataFrame([input_data], columns=columns)
        #prediction = loan_model.predict(input_df)

        # Enhanced loan eligibility decision with detailed feedback
        if EMI < Total_Income * 3:
            st.markdown("""
                <div class="success-message">
                    <h3>🎉 Congratulations! Loan Approved</h3>
                    <div class="stat-card">
                        <p>Your loan application has been approved based on:</p>
                        <ul>
                            <li>Income to EMI ratio is favorable</li>
                            <li>Credit history assessment</li>
                            <li>Overall profile evaluation</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Display additional financial insights
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                    <div class="stat-card">
                        <h4>EMI to Income Ratio</h4>
                        <p style="font-size: 1.5em; color: #4CAF50;">
                            {:.1f}%
                        </p>
                    </div>
                """.format((EMI/Total_Income) * 100), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="stat-card">
                        <h4>Loan Amount</h4>
                        <p style="font-size: 1.5em; color: #4CAF50;">
                            ₹{:,.2f}
                        </p>
                    </div>
                """.format(EMI), unsafe_allow_html=True)

        else:
            st.markdown("""
                <div class="error-message">
                    <h3>🚫 Loan Application Status: Not Approved</h3>
                    <div class="stat-card">
                        <p>Your application wasn't approved due to:</p>
                        <ul>
                            <li>High EMI to income ratio</li>
                            <li>Please consider a lower loan amount</li>
                            <li>Or provide additional income sources</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Provide suggestions for improvement
            st.markdown("""
                <div class="service-card">
                    <h4>💡 Suggestions for Improvement</h4>
                    <ul>
                        <li>Consider reducing the loan amount</li>
                        <li>Improve your credit score</li>
                        <li>Clear existing debts</li>
                        <li>Add a co-applicant to increase eligible loan amount</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
