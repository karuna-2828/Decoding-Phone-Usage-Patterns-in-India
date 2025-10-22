import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set page configuration
st.set_page_config(
    page_title="Phone Usage Analysis Dashboard",
    page_icon="📱",
    layout="wide"
)

# Define global usage categories
USAGE_CATEGORIES = {
    0: "Social Media",
    1: "Gaming",
    2: "Streaming",
    3: "Professional",
    4: "Educational",
    5: "Communication"
}

# Main title
st.title("📱 Phone Usage Analysis Dashboard")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("decision_tree_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'decision_tree_model.pkl' not found. Please ensure it exists in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('clustering_results.csv')
        return df
    except FileNotFoundError:
        st.error("Data file 'clustering_results.csv' not found. Please ensure it exists in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_cluster_plot(df, cluster_col):
    # Use plt.style.use('seaborn-v0_8') instead of 'seaborn'
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create color palette for better visibility
    n_clusters = len(df[cluster_col].unique())
    palette = sns.color_palette("husl", n_clusters)
    
    # Create scatter plot with enhanced styling
    sns.scatterplot(data=df, 
                   x='Screen Time (hrs/day)', 
                   y='Data Usage (GB/month)', 
                   hue=cluster_col,
                   palette=palette,
                   alpha=0.6)
    
    # Enhance plot styling
    plt.title('User Segments by Screen Time and Data Usage', pad=20)
    plt.xlabel('Screen Time (hours/day)', labelpad=10)
    plt.ylabel('Data Usage (GB/month)', labelpad=10)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return fig

# Load the model and data
dec_tree_model = load_model()
df = load_data()

if df is not None:
    cluster_columns = [col for col in df.columns if col.startswith('cluster_')]
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ["Overview", "Usage Patterns", "User Classification", "User Segmentation"]
    )

    if page == "Overview":
        st.header("📊 Dashboard Overview")
        st.write("Welcome to the Phone Usage Analysis Dashboard. This tool helps analyze and predict user behavior patterns.")
        

        col1, col2 = st.columns(2)
        with col1:
                st.metric("Total Users", len(df))
                st.metric("Average Screen Time", f"{df['Screen Time (hrs/day)'].mean():.1f} hrs/day")
        with col2:
                st.metric("Average Data Usage", f"{df['Data Usage (GB/month)'].mean():.1f} GB/month")
                
                # Enhanced most common use calculation with error handling

                df['Primary Use'] = df['Primary Use'].astype(int)

                try:
                    use_counts = df['Primary Use'].value_counts()
                    if len(use_counts) > 0:
                        most_common = use_counts.index[0]
                        most_common = USAGE_CATEGORIES.get(int(most_common), f"Category {most_common}")
                        
                    else:
                        most_common = "No data available"
                    
                                   
                except Exception as e:
                    st.error(f"Error calculating most common use: {str(e)}")
                    most_common = "Error in calculation"
                
                st.metric("Most Common Use", most_common)

    elif page == "Usage Patterns":
        st.header("📱 Usage Patterns")
        if df is not None:
            # Add usage pattern visualizations here
            st.subheader("Daily Usage Distribution")
            
            # Create enhanced histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='Screen Time (hrs/day)', bins=20, color='skyblue')
            plt.title('Distribution of Daily Screen Time', pad=20)
            plt.xlabel('Screen Time (hours/day)', labelpad=10)
            plt.ylabel('Count', labelpad=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close()

            st.subheader("📊 Select Feature for Visualization")

            # List of features to visualize
            feature_options = [
                "Screen Time (hrs/day)", "Social Media Time (hrs/day)", 
                "Streaming Time (hrs/day)", "Gaming Time (hrs/day)", 
                "Data Usage (GB/month)", "E-commerce Spend (INR/month)", 
                "Monthly Recharge Cost (INR)"
            ]

            # Let user choose a feature
            selected_feature = st.selectbox("Choose a feature to visualize", feature_options)

            st.subheader(f"📊 Distribution of {selected_feature}")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax)

            plt.xlabel(selected_feature)
            plt.ylabel("Count")
            plt.title(f"Distribution of {selected_feature}")
            st.pyplot(fig)
            
    elif page == "User Classification":
        st.header("🎯 User Classification")
        
        if dec_tree_model is not None:
            st.subheader("Predict Primary Use")
            
            with st.form("classification_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Age", min_value=18, max_value=80, value=25)
                    screen_time = st.number_input("Screen Time (hrs/day)", 
                                                min_value=0.5, max_value=24.0, value=4.0, step=0.5)
                    data_usage = st.number_input("Data Usage (GB/month)", 
                                               min_value=1.0, value=30.0, step=1.0)
                    calls_duration = st.number_input("Calls Duration (mins/day)", 
                                                   min_value=1.0, value=45.0, step=5.0)
                
                with col2:
                    num_apps = st.number_input("Number of Apps Installed", 
                                             min_value=5, value=25, step=1)
                    social_media = st.number_input("Social Media Time (hrs/day)", 
                                                 min_value=0.0, max_value=24.0, value=2.5, step=0.5)
                    ecommerce_spend = st.number_input("E-commerce Spend (INR/month)", 
                                                    min_value=0.0, value=2000.0, step=100.0)
                    streaming_time = st.number_input("Streaming Time (hrs/day)", 
                                                   min_value=0.0, max_value=24.0, value=1.5, step=0.5)
                
                with col3:
                    gaming_time = st.number_input("Gaming Time (hrs/day)", 
                                                min_value=0.0, max_value=24.0, value=1.0, step=0.5)
                    monthly_cost = st.number_input("Monthly Recharge Cost (INR)", 
                                                 min_value=200.0, value=699.0, step=50.0)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    os = st.selectbox("Operating System", ["Android", "iOS"])
                
                submit_button = st.form_submit_button("Predict Primary Use")
            
            if submit_button:
                total_time = screen_time + social_media + gaming_time + streaming_time
                if total_time > 24:
                    st.error("❌ Total time across activities exceeds 24 hours. Please adjust your inputs.")
                else:
                    try:
                        # Create input DataFrame
                        input_data = pd.DataFrame([[
                            monthly_cost, gaming_time, streaming_time, ecommerce_spend, 
                            social_media, num_apps, calls_duration, data_usage, 
                            screen_time, age
                        ]], columns=[
                            "Monthly Recharge Cost (INR)", "Gaming Time (hrs/day)", "Streaming Time (hrs/day)", 
                            "E-commerce Spend (INR/month)", "Social Media Time (hrs/day)", "Number of Apps Installed", 
                            "Calls Duration (mins/day)", "Data Usage (GB/month)", "Screen Time (hrs/day)", "Age"
                        ])
                        
                        prediction_num =dec_tree_model.predict(input_data)[0]
                        prediction_category = USAGE_CATEGORIES.get(prediction_num, "Unknown Category")
                        st.success(f"🎯 Predicted Primary Use: {prediction_category}")
                    
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                        st.info("ℹ️ Please ensure all input values are valid.")

    elif page == "User Segmentation":
         st.header("👥 User Segmentation")
        
         if cluster_columns:
            # Display cluster analysis
            st.subheader("User Clusters")
            
            # Cluster selection
            cluster_col = st.selectbox("Select Cluster Type", cluster_columns)
            
            # Create and display scatter plot
            fig = create_cluster_plot(df, cluster_col)
            st.pyplot(fig)
            plt.close()
            USAGE_CATEGORIES = {
                0: "Social Media",
                1: "Gaming",
                2: "Streaming",
                3: "Professional",
                4: "Educational"
}
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            for cluster in sorted(df[cluster_col].unique()):
                cluster_data = df[df[cluster_col] == cluster]
                with st.expander(f"Cluster {cluster}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Users", len(cluster_data))
                        st.metric("Avg Screen Time", f"{cluster_data['Screen Time (hrs/day)'].mean():.1f} hrs/day")
                    with col2:
                        st.metric("Avg Data Usage", f"{cluster_data['Data Usage (GB/month)'].mean():.1f} GB/month")
                        
                        # Convert Primary Use to int & get most common use
                        most_common_use_num = cluster_data['Primary Use'].astype(int).mode()[0]
                        most_common_use = USAGE_CATEGORIES.get(most_common_use_num, f"Category {most_common_use_num}")

                        st.metric("Most Common Use", most_common_use)
                        
                        
         else:
            st.warning("⚠️ No cluster columns found in the dataset.")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit • Data analysis and visualization dashboard</p>
    </div>
    """, unsafe_allow_html=True)