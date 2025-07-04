import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('titanic_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, encoders

@st.cache_data
def load_titanic_data():
    return pd.read_csv('train.csv')

# Load resources
model, encoders = load_model_and_encoders()
titanic_data = load_titanic_data()

# Main title
st.title("🚢 Titanic Survival Prediction")
st.markdown("**Predict passenger survival on the RMS Titanic using machine learning**")

# Sidebar for user inputs
st.sidebar.header("👤 Passenger Information")

# Input widgets
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], 
                             help="1st = Upper class, 2nd = Middle class, 3rd = Lower class")

sex = st.sidebar.selectbox("Gender", ["male", "female"])

age = st.sidebar.slider("Age", 0, 80, 30)

sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)

parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)

fare = st.sidebar.number_input("Fare (£)", 0.0, 500.0, 32.0, step=0.1)

embarked = st.sidebar.selectbox("Port of Embarkation", 
                                ["Southampton", "Cherbourg", "Queenstown"])

# Map embarked to codes
embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
embarked_code = embarked_map[embarked]

# Feature engineering for prediction
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Determine title based on age and gender
if age < 18:
    title = "Master" if sex == "male" else "Miss"
elif sex == "female":
    title = "Mrs" if age > 25 else "Miss"
else:
    title = "Mr"

# Age group
if age <= 12:
    age_group = "Child"
elif age <= 18:
    age_group = "Teen"
elif age <= 35:
    age_group = "Adult"
elif age <= 60:
    age_group = "Middle"
else:
    age_group = "Senior"

# Fare group (simplified)
if fare <= 7.91:
    fare_group = "Low"
elif fare <= 14.45:
    fare_group = "Medium"
elif fare <= 31.0:
    fare_group = "High"
else:
    fare_group = "Very High"

# Create input dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [encoders['Sex'].transform([sex])[0]],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [encoders['Embarked'].transform([embarked_code])[0]],
    'FamilySize': [family_size],
    'IsAlone': [is_alone],
    'Title': [encoders['Title'].transform([title])[0]],
    'AgeGroup': [encoders['AgeGroup'].transform([age_group])[0]],
    'FareGroup': [encoders['FareGroup'].transform([fare_group])[0]]
})

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Prediction Results")
    
    # Display prediction with styling
    if prediction == 1:
        st.success(f"**SURVIVED** 🎉")
        st.markdown(f"**Survival Probability: {probability[1]:.1%}**")
    else:
        st.error(f"**DID NOT SURVIVE** 😢")
        st.markdown(f"**Survival Probability: {probability[1]:.1%}**")
    
    # Probability visualization
    fig_prob = go.Figure(go.Bar(
        x=['Did Not Survive', 'Survived'],
        y=[probability[0], probability[1]],
        marker_color=['#ff6b6b', '#51cf66'],
        text=[f'{probability[0]:.1%}', f'{probability[1]:.1%}'],
        textposition='auto',
    ))
    fig_prob.update_layout(
        title="Survival Probability",
        yaxis_title="Probability",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col2:
    st.subheader("📊 Passenger Profile")
    
    # Display passenger information
    st.markdown(f"""
    **Class:** {pclass} {'(First)' if pclass==1 else '(Second)' if pclass==2 else '(Third)'}
    
    **Gender:** {sex.title()}
    
    **Age:** {age} years ({age_group})
    
    **Family Size:** {family_size}
    
    **Fare:** £{fare:.2f} ({fare_group})
    
    **Embarked:** {embarked}
    
    **Title:** {title}
    """)

# Historical context and visualizations
st.subheader("📈 Historical Data Analysis")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Survival by Class & Gender", "Age Distribution", "Fare Analysis"])

with tab1:
    # Survival by class and gender
    survival_by_class_sex = titanic_data.groupby(['Pclass', 'Sex'])['Survived'].agg(['count', 'sum']).reset_index()
    survival_by_class_sex['survival_rate'] = survival_by_class_sex['sum'] / survival_by_class_sex['count']
    
    fig_class_sex = px.bar(survival_by_class_sex, 
                          x='Pclass', y='survival_rate', 
                          color='Sex', barmode='group',
                          title="Survival Rate by Passenger Class and Gender",
                          labels={'survival_rate': 'Survival Rate', 'Pclass': 'Passenger Class'})
    st.plotly_chart(fig_class_sex, use_container_width=True)

with tab2:
    # Age distribution
    fig_age = make_subplots(rows=1, cols=2, subplot_titles=('Survivors', 'Non-survivors'))
    
    survivors = titanic_data[titanic_data['Survived'] == 1]['Age'].dropna()
    non_survivors = titanic_data[titanic_data['Survived'] == 0]['Age'].dropna()
    
    fig_age.add_trace(go.Histogram(x=survivors, name='Survived', marker_color='#51cf66'), row=1, col=1)
    fig_age.add_trace(go.Histogram(x=non_survivors, name='Did Not Survive', marker_color='#ff6b6b'), row=1, col=2)
    
    fig_age.update_layout(title="Age Distribution by Survival Status", showlegend=False)
    st.plotly_chart(fig_age, use_container_width=True)

with tab3:
    # Fare analysis
    fig_fare = px.box(titanic_data, x='Survived', y='Fare', 
                     title="Fare Distribution by Survival Status",
                     labels={'Survived': 'Survived (0=No, 1=Yes)', 'Fare': 'Fare (£)'})
    st.plotly_chart(fig_fare, use_container_width=True)

# Feature importance
st.subheader("🔍 Model Insights")
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
importance = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=True)

fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                       orientation='h', title="Feature Importance in Survival Prediction")
st.plotly_chart(fig_importance, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**About the Titanic Dataset:** The RMS Titanic sank on April 15, 1912, during her maiden voyage. This model analyzes passenger data to predict survival likelihood based on various factors including class, gender, age, and family relationships.")
