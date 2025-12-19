import plotly.express as px
from src.utils import load_model
from src.config import TUNED_MODEL_PATH, BASE_MODEL_PATH, FINAL_FEATURES
import os
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="TATA Steel - Machine Failure Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .failure-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .no-failure-risk {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_model(model_path):
    """Load the trained model"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_feature_importance_plot(model, feature_names):
    """Create an interactive feature importance plot"""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    return fig

def main():
    # Header
    st.markdown(
        '<p class="main-header">🏭 TATA Steel Machine Failure Prediction System</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Sidebar - Model Selection
    st.sidebar.header("⚙️ Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Tuned LightGBM (Recommended)", "Base LightGBM"],
        help="Choose between the hyperparameter-tuned model or the base model"
    )
    
    model_path = TUNED_MODEL_PATH if "Tuned" in model_choice else BASE_MODEL_PATH
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_prediction_model(model_path)
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure the model file exists.")
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Sidebar - Information
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        """
        This application predicts machine failures in TATA Steel's 
        production facilities using real-time operational data.
        
        **Key Features:**
        - Real-time failure prediction
        - Binary failure decision
        - Feature importance analysis
        - Batch prediction support
        """
    )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Insights"]
    )
    
    # ===================================================================
    # TAB 1: SINGLE PREDICTION
    # ===================================================================
    with tab1:
        st.header("Single Machine Prediction")
        st.markdown(
            "Enter the operational parameters below to predict machine failure."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Process Parameters")
            
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=10.0,
                max_value=1000.0,
                value=300.0,
                step=0.1
            )
            
            rotation_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=100,
                max_value=6000,
                value=1500,
                step=10
            )
            
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.5
            )
            
            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=500,
                value=100,
                step=1
            )
        
        with col2:
            st.subheader("Product & Failure Indicators")
            
            product_type = st.selectbox(
                "Product Type",
                ["L (Low)", "M (Medium)", "H (High)"]
            )
            type_encoded = {'L (Low)': 1, 'M (Medium)': 2, 'H (High)': 3}[product_type]
            
            st.markdown("**Previous Failure Indicators:**")
            twf = st.checkbox("Tool Wear Failure (TWF)", value=False)
            hdf = st.checkbox("Heat Dissipation Failure (HDF)", value=False)
            pwf = st.checkbox("Power Failure (PWF)", value=False)
            osf = st.checkbox("Overstrain Failure (OSF)", value=False)
            rnf = st.checkbox("Random Failure (RNF)", value=False)
        
        st.markdown("---")
        
        if st.button(
            "🔍 Predict Failure",
            type="primary",
            use_container_width=True
        ):
            input_data = pd.DataFrame({
                'Air_temperature_K_': [air_temp],
                'Rotational_speed_rpm_': [rotation_speed],
                'Torque_Nm_': [torque],
                'Tool_wear_min_': [tool_wear],
                'TWF': [int(twf)],
                'HDF': [int(hdf)],
                'PWF': [int(pwf)],
                'OSF': [int(osf)],
                'RNF': [int(rnf)],
                'Type_encoded': [type_encoded]
            })
            
            with st.spinner("Analyzing machine condition..."):
                prediction = model.predict(input_data)[0]
            
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-box failure-risk">'
                    '<h2 style="color:#d32f2f;">❌ MACHINE FAILURE PREDICTED</h2>'
                    '<p style="font-size:1.2rem;">Immediate maintenance required</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-box no-failure-risk">'
                    '<h2 style="color:#388e3c;">✅ NO FAILURE PREDICTED</h2>'
                    '<p style="font-size:1.2rem;">Machine operating normally</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
    
    # ===================================================================
    # TAB 2: BATCH PREDICTION
    # ===================================================================
    with tab2:
        st.header("Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.dataframe(df_batch.head(), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                predictions = model.predict(df_batch[FINAL_FEATURES])
                df_batch['Prediction'] = predictions
                df_batch['Failure_Status'] = df_batch['Prediction'].map(
                    {1: 'Failure', 0: 'No Failure'}
                )
                
                st.dataframe(df_batch, use_container_width=True)
    
    # ===================================================================
    # TAB 3: MODEL INSIGHTS
    # ===================================================================
    with tab3:
        st.header("Model Insights")
        fig = create_feature_importance_plot(model, FINAL_FEATURES)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
