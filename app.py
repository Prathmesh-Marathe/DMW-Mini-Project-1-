import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from google import genai
from google.genai import types
from googletrans import Translator
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# EARLY SESSION STATE INITIALIZATION
# =======================================================
if 'target_language_code' not in st.session_state:
    st.session_state.target_language_code = 'en'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'performance_features' not in st.session_state:
    st.session_state.performance_features = []
if 'uploaded_file_name_state' not in st.session_state:
    st.session_state.uploaded_file_name_state = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = "N/A"
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'class_report' not in st.session_state:
    st.session_state.class_report = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# --- PREDICTION INPUT FEATURES FOR STUDENT PERFORMANCE ---
PREDICTION_INPUT_FEATURES = [
    'gender', 'class', 'age', 'attendance_percentage',
    'weight', 'height', 'health_score', 'math_prev3', 'math_prev2',
    'math_prev1', 'science_prev2', 'science_prev1', 'english_prev3',
    'english_prev2', 'english_prev1'
]

# ===========================
# Language Configuration
# ===========================
LANGUAGE_MAP = {
    'English': 'en',
    'Hindi': 'hi',
    'Spanish': 'es',
    'French': 'fr',
    'Chinese (Simplified)': 'zh-cn',
    'Marathi': 'mr'  
}

TRANSLATION_DICT = {
    "app_title": " Chorachiwadi Student Performance Analysis System",
    "app_subtitle": "Empowering Teachers with Data-Driven Insights for Better Learning Outcomes",
    "upload_file_title": "Upload your dataset (CSV)",
    "upload_warning": "Please upload a Student dataset to continue",
    "home_welcome": "Welcome to Student Performance Analysis System",
    "home_analyze": "Analyze and visualize student performance data.",
    "home_detect": "Predict student performance trends.",
    "home_cluster": "Group students based on performance patterns.",
    "home_chat": "Get instant advice from the AI Education Assistant.",
    "home_decisions": "Make smart decisions for improved learning outcomes.",
    "home_tip": "Tip: Navigate through the menu to explore data analysis features.",
    "data_overview_header": "Dataset Overview",
    "filters_header": "Data Insights & Analysis",
    "cluster_header": "Student Performance Groups",
    "classification_header": "Performance Prediction",
    "graphs_header": "Visual Analysis",
    "chatbot_header": "AI Education Assistant",
    "apriori_header": "Pattern Discovery in Student Data",
    "sidebar_language": "Select Language",
    "Navigation Menu": "Navigation Menu",
    "Mean": "Average", "Sum": "Total", "Max": "Highest", "Min": "Lowest", "Count": "Count",
    "Predicted": "Predicted", "True": "True", "Actual": "Actual", "Frequency": "Frequency",
    "Feature": "Feature", "Chi-Squared Score": "Importance Score",
    "Select Graph Type": "Choose Visualization Type",
    "Line": "Trend Line", "Bar": "Bar Chart", "Scatter": "Scatter Plot", "Histogram": "Distribution",
    "Select X-axis (Categorical)": "Choose Category",
    "Select Y-axis (Numeric Measure)": "Choose What to Measure",
    "Select Aggregation": "How to Calculate",
    "Generate Graph": "Show Visualization",
    "Select Column (Numeric)": "Select Data to Analyze",
    "Number of Bins": "Number of Groups",
    "Select Columns (X, Y)": "Select Data Columns",
    "Please select both an X-axis and a Y-axis.": "Please select both category and measure.",
    "Please select a column for the Histogram.": "Please select a column to analyze.",
    "Select at least two columns for Line/Scatter plot.": "Select at least two data columns.",
    "The AI Education Assistant is thinking...": "The AI Education Assistant is thinking...",
    "Ask about student performance, teaching strategies, or educational questions...": "Ask about student performance, teaching strategies, or educational questions...",
    "Please upload a dataset first.": "Please upload a dataset first.",
    "Please upload a dataset first in the 'Data Upload' section.": "Please upload a dataset first in the 'Upload Data' section.",
}

def T(key):
    text = TRANSLATION_DICT.get(key, key)
    if st.session_state.target_language_code == 'en':
        return text
    try:
        translator = Translator()
        return translator.translate(text, dest=st.session_state.target_language_code).text
    except Exception:
        return text

# ===========================
# Gemini AI Setup
# ===========================
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if "GEMINI_API_KEY" not in st.secrets:
        return None
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None

client = get_gemini_client()
GEMINI_MODEL = "gemini-2.5-flash"

# ===========================
# Streamlit Page Setup
# ===========================
st.set_page_config(
    page_title=T("app_title"),
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# Modern Gradient Theme Styling
# ===========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFFFFF; 
        font-family: 'Inter', sans-serif; 
    }
    
    h1, h2, h3, h4 { 
        color: #FFFFFF; 
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        color: #FFFFFF;
        border-right: 2px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] * { 
        color: #FFFFFF !important; 
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: #FFFFFF;
        border-radius: 25px; 
        padding: 0.8em 2em;
        font-size: 1em; 
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
    }
    
    div.stButton > button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.6);
    }
    
    .stDataFrame { 
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        padding: 10px;
    }
    
    div[data-testid="stMetricValue"] { 
        color: #f093fb; 
        font-weight: 700;
        font-size: 2em;
    }
    
    .insight-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .insight-title {
        color: #f093fb;
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-left: 4px solid #f093fb;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        color: #FFFFFF;
    }
    
    .stSelectbox, .stMultiSelect, .stSlider {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Custom metric styling */
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9em;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 15px 25px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #f093fb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================
# Helper Functions
# ===========================
def categorize_performance(score):
    """Categorize student based on score"""
    if score >= 75:
        return 'Excellent'
    elif score >= 50:
        return 'Good'
    else:
        return 'Needs Improvement'

def categorize_pass_fail(score):
    """Categorize pass/fail based on score"""
    return 'Pass' if score >= 40 else 'Fail'

def categorize_attendance(attendance):
    """Categorize attendance"""
    if attendance >= 90:
        return 'Outstanding'
    elif attendance >= 75:
        return 'Good'
    elif attendance >= 60:
        return 'Acceptable'
    else:
        return 'Poor'

def auto_preprocess_data(df, for_training=False):
    """Automatically clean and prepare data without user intervention"""
    df_processed = df.copy()
    
    # Handle numeric columns
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        df_processed[numeric_cols] = imputer_num.fit_transform(df_processed[numeric_cols])
    
    # Handle categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('Unknown').astype(str)
        if for_training:
            le = LabelEncoder()
            le.fit(df_processed[col])
            df_processed[col] = le.transform(df_processed[col])
            st.session_state.label_encoders[col] = le
        elif col in st.session_state.label_encoders:
            le = st.session_state.label_encoders[col]
            def safe_transform(x):
                try:
                    return le.transform([x])[0]
                except ValueError:
                    if 'Unknown' in le.classes_:
                        return le.transform(['Unknown'])[0]
                    return -1
            df_processed[col] = df_processed[col].apply(safe_transform)
        else:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            st.session_state.label_encoders[col] = le
    
    return df_processed

# ===========================
# Model Training Function
# ===========================
@st.cache_data(show_spinner='🤖 Training AI Model...')
def train_performance_model(df, features, target='performance_category'):
    try:
        if target not in df.columns:
            raise ValueError(f"No '{target}' column found for prediction.")
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError("No suitable features found for training the model.")
        
        temp_df = df.copy()
        target_le = LabelEncoder()
        temp_df['target_encoded'] = target_le.fit_transform(
            temp_df[target].astype(str).fillna('Unknown'))
        st.session_state.label_encoders[target] = target_le
        
        status_text = st.empty()
        status_text.text("🔧 Preparing data for AI model...")
        
        df_features = temp_df[available_features].copy()
        df_processed = auto_preprocess_data(df_features, for_training=True)
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(df_processed)
        poly_feature_names = poly.get_feature_names_out(df_processed.columns)
        X_poly = pd.DataFrame(X_poly, columns=poly_feature_names)
        
        y = temp_df['target_encoded']
        
        k_best = min(20, X_poly.shape[1])
        selector = SelectKBest(chi2, k=k_best)
        X_new = selector.fit_transform(X_poly, y)
        selected_features_mask = selector.get_support()
        X_selected_features = X_poly.columns[selected_features_mask].tolist()
        X = pd.DataFrame(X_new, columns=X_selected_features)
        
        st.session_state.poly = poly
        st.session_state.selector = selector
        st.session_state.performance_features = X_selected_features
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state.scaler = scaler
        
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train)
        except ValueError:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
        status_text.text("🎯 Training AI model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_train_resampled, y_train_resampled)
        
        st.session_state.model_performance = rf_model
        st.session_state.best_model_name = 'AI Predictor'
        
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred, output_dict=True)
        
        st.session_state.accuracy = accuracy
        st.session_state.class_report = class_rep
        
        status_text.text("✅ AI model ready!")
        return y_pred
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.session_state.model_performance = None
        st.session_state.performance_features = []
        st.session_state.best_model_name = "N/A"
        st.session_state.accuracy = None
        st.session_state.class_report = None
        return None
    finally:
        status_text.empty()

# ===========================
# Performance Classification
# ===========================
def classify_performance(input_data, return_probabilities=False):
    try:
        if st.session_state.model_performance is None:
            return None
        
        input_df = pd.DataFrame([input_data])
        required_features = st.session_state.performance_features
        base_features = st.session_state.poly.feature_names_in_
        
        input_for_prep = input_df.copy()
        
        for feat in base_features:
            if feat not in input_for_prep.columns:
                if st.session_state.df is not None and feat in st.session_state.df.columns:
                    dtype = st.session_state.df[feat].dtype
                    if dtype in ['int64', 'float64']:
                        input_for_prep[feat] = st.session_state.df[feat].mean()
                    else:
                        mode_val = st.session_state.df[feat].mode()
                        input_for_prep[feat] = mode_val[0] if not mode_val.empty else 'Unknown'
                else:
                    input_for_prep[feat] = 0.0 if st.session_state.df is None else 'Unknown'

        input_processed = auto_preprocess_data(input_for_prep[base_features].copy())
        input_poly = st.session_state.poly.transform(input_processed)
        input_poly = pd.DataFrame(
            input_poly,
            columns=st.session_state.poly.get_feature_names_out(base_features)
        )
        
        input_features = input_poly[required_features]
        input_features_scaled = st.session_state.scaler.transform(input_features)
        
        model = st.session_state.model_performance
        predicted_class_idx = model.predict(input_features_scaled)[0]
        
        le = st.session_state.label_encoders.get('performance_category')
        if le:
            predicted_performance = le.inverse_transform([predicted_class_idx])[0]
            
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_features_scaled)[0]
                performance_probabilities = {
                    le.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
                return predicted_performance, performance_probabilities
            
            return predicted_performance
        
        return predicted_class_idx
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# ===========================
# Data Analysis Operations
# ===========================
def perform_data_analysis(df, operation, dimensions, measures):
    df_temp = df.copy()
    numeric_measures = [m for m in measures if m in df_temp.select_dtypes(include=np.number).columns]
    
    if operation in ['filter_by_category', 'compare_groups'] and 'slice_dimension' in st.session_state and 'slice_values' in st.session_state:
        slice_dim = st.session_state.slice_dimension
        slice_vals = st.session_state.slice_values
        if slice_dim in df_temp.columns and slice_vals:
            df_temp = df_temp[df_temp[slice_dim].isin(slice_vals)]
            if operation == 'filter_by_category':
                return df_temp.groupby(dimensions)[numeric_measures].mean().reset_index(), df_temp 

    try:
        if operation in ['filter_by_category', 'compare_groups']:
            if dimensions and numeric_measures:
                return df_temp.groupby(dimensions)[numeric_measures].mean().reset_index(), df_temp 
            elif dimensions:
                return df_temp.groupby(dimensions).size().to_frame(name='Count').reset_index(), df_temp
            else:
                return df_temp[numeric_measures].mean().to_frame().T, df_temp

        elif operation == 'cross_analyze':
            if len(dimensions) >= 2 and numeric_measures:
                pivot_df = df_temp.pivot_table(
                    index=dimensions[0], columns=dimensions[1],
                    values=numeric_measures[0], aggfunc='mean', fill_value=0)
                return pivot_df, df_temp
            return None, df_temp
            
        else:
            return None, df_temp
            
    except Exception as e:
        st.error(f"Error performing analysis: {str(e)}")
        return None, None

# ===========================
# Clustering
# ===========================
def perform_clustering(df, selected_features, n_clusters=3):
    try:
        X = df[selected_features].copy()
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Group'] = clusters
        return df_clustered
    except Exception as e:
        st.error(f"Error creating groups: {str(e)}")
        return None

# ===========================
# Pattern Discovery (Apriori)
# ===========================
def discover_patterns(df):
    """Discover patterns in student data"""
    try:
        df_analysis = df.copy()
        
        df_analysis['attendance_category'] = df_analysis['attendance_percentage'].apply(categorize_attendance)
        
        score_cols = [col for col in df_analysis.columns if 'math' in col.lower() or 'science' in col.lower() or 'english' in col.lower()]
        if score_cols:
            df_analysis['avg_score'] = df_analysis[score_cols].mean(axis=1)
            df_analysis['performance_category'] = df_analysis['avg_score'].apply(categorize_performance)
            df_analysis['pass_fail'] = df_analysis['avg_score'].apply(categorize_pass_fail)
        
        transactions = []
        for idx, row in df_analysis.iterrows():
            transaction = []
            transaction.append(f"Attendance_{row['attendance_category']}")
            if 'performance_category' in row:
                transaction.append(f"Performance_{row['performance_category']}")
            if 'pass_fail' in row:
                transaction.append(f"Result_{row['pass_fail']}")
            if 'gender' in row:
                transaction.append(f"Gender_{row['gender']}")
            if 'class' in row:
                transaction.append(f"Class_{row['class']}")
            transactions.append(transaction)
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            
            return rules, frequent_itemsets
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error discovering patterns: {str(e)}")
        return None, None

# ===========================
# Insight Generator
# ===========================
def generate_insights(graph_type, df, x_col=None, y_col=None, agg_func=None):
    """Generate insights for visualizations"""
    insights = []
    
    try:
        if graph_type == "Bar Chart" and x_col and y_col:
            if agg_func == 'mean':
                agg_data = df.groupby(x_col)[y_col].mean()
                max_cat = agg_data.idxmax()
                min_cat = agg_data.idxmin()
                insights.append(f"📊 {x_col.title().replace('_', ' ')} '{max_cat}' shows the best performance with {agg_data.max():.2f}")
                insights.append(f"📉 {x_col.title().replace('_', ' ')} '{min_cat}' needs attention with {agg_data.min():.2f}")
                insights.append(f"📈 Performance gap is {(agg_data.max() - agg_data.min()):.2f} points")
                
                if 'class' in x_col.lower():
                    insights.append(f"🎓 Different classes show varying performance - consider tailored teaching approaches")
                if 'gender' in x_col.lower():
                    insights.append(f"👥 Gender-based patterns can help identify specific support needs")
            
        elif graph_type == "Distribution" and x_col:
            mean_val = df[x_col].mean()
            median_val = df[x_col].median()
            std_val = df[x_col].std()
            insights.append(f"📊 Average {x_col.replace('_', ' ').title()}: {mean_val:.2f}")
            insights.append(f"📍 Middle Value: {median_val:.2f}")
            insights.append(f"📏 Variation: {std_val:.2f}")
            
            if mean_val > median_val:
                insights.append(f"⚠️ Some high performers are pulling the average up")
            elif mean_val < median_val:
                insights.append(f"✅ Most students perform above the average")
            else:
                insights.append(f"✅ Performance is well-balanced across students")
            
            if 'attendance' in x_col.lower():
                if mean_val >= 85:
                    insights.append(f"✅ Excellent attendance overall - keep it up!")
                elif mean_val >= 75:
                    insights.append(f"⚠️ Attendance could be better - consider engagement programs")
                else:
                    insights.append(f"🚨 Low attendance needs immediate attention")
                    
        elif graph_type in ["Scatter Plot", "Trend Line"] and x_col and y_col:
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            insights.append(f"📊 Connection strength between {x_col.replace('_', ' ')} and {y_col.replace('_', ' ')}: {abs(correlation):.2f}")
            
            if abs(correlation) > 0.7:
                insights.append(f"💪 Strong relationship - they move together")
                if correlation > 0:
                    insights.append(f"📈 When {x_col.replace('_', ' ')} increases, {y_col.replace('_', ' ')} tends to increase too")
                else:
                    insights.append(f"📉 When {x_col.replace('_', ' ')} increases, {y_col.replace('_', ' ')} tends to decrease")
            elif abs(correlation) > 0.4:
                insights.append(f"📊 Moderate relationship detected")
            else:
                insights.append(f"📉 Weak relationship - these factors may be independent")
            
            if 'attendance' in x_col.lower() or 'attendance' in y_col.lower():
                if abs(correlation) > 0.5:
                    insights.append(f"🎯 Attendance has a significant impact on performance")
                    
    except Exception as e:
        insights.append(f"Unable to generate detailed insights: {str(e)}")
    
    return insights

# ===========================
# Sidebar Navigation
# ===========================
selected_lang_name = st.sidebar.selectbox(
    "🌐 " + T("sidebar_language"),
    options=list(LANGUAGE_MAP.keys()),
    key='language_selector_widget'
)
st.session_state.target_language_code = LANGUAGE_MAP[selected_lang_name]

st.sidebar.markdown("---")

menu_options = [
    "🏠 Home",
    "📤 Upload Data",
    "📊 Data Insights",
    "👥 Student Groups",
    "🎯 Performance Prediction",
    "📈 Visual Analysis",
    "🔍 Pattern Discovery",
    "💬 AI Assistant"
]
selected = st.sidebar.radio("🧭 " + T("Navigation Menu"), menu_options)

# ===========================
# Home Page
# ===========================
if selected == "🏠 Home":
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 3em 0;'>
            <h1 style='font-size: 3.5em; margin-bottom: 0.2em;'>🎓</h1>
            <h1 style='font-size: 2.5em; margin-bottom: 0.5em;'>Chorachiwadi - Student Success Tracker</h1>
            <p style='font-size: 1.2em; opacity: 0.9;'>Understand Your Students Better with Smart Analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='insight-card'>
                <h3 style='text-align: center;'>📊 Smart Insights</h3>
                <p style='text-align: center; opacity: 0.8;'>Get clear, actionable insights about student performance and trends</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='insight-card'>
                <h3 style='text-align: center;'>🎯 Predict Success</h3>
                <p style='text-align: center; opacity: 0.8;'>AI-powered predictions to identify students who need help early</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='insight-card'>
                <h3 style='text-align: center;'>💡 Find Patterns</h3>
                <p style='text-align: center; opacity: 0.8;'>Discover what factors influence student success the most</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
        <div class='info-box'>
            <h3>🚀 Quick Start Guide:</h3>
            <ol style='margin-top: 10px; line-height: 2;'>
                <li><strong>Upload Data:</strong> Start by uploading your student records (Excel or CSV file)</li>
                <li><strong>Explore Insights:</strong> View charts and understand performance patterns</li>
                <li><strong>Create Groups:</strong> Automatically group students with similar characteristics</li>
                <li><strong>Predict Outcomes:</strong> Use AI to predict student performance</li>
                <li><strong>Get Recommendations:</strong> Chat with AI assistant for personalized advice</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        st.success(f"✅ Data loaded successfully! You have {len(st.session_state.df)} student records ready to analyze.")
    else:
        st.info("👆 Click on '📤 Upload Data' in the sidebar to get started!")

# ===========================
# Upload Data
# ===========================
elif selected == "📤 Upload Data":
    st.markdown("<div class='section-header'><h2>📤 Upload Your Student Data</h2></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <p><strong>What you need:</strong> A file containing student information like names, scores, attendance, etc.</p>
            <p><strong>Supported formats:</strong> CSV files (.csv)</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose your file", type=["csv"], help="Upload a CSV file with student data")
    
    if uploaded_file is not None:
        try:
            current_file_name = uploaded_file.name
            if st.session_state.uploaded_file_name_state != current_file_name:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.lower()
                
                # Automatic preprocessing
                with st.spinner("🔧 Preparing your data..."):
                    # Remove duplicates
                    df = df.drop_duplicates()
                    
                    # Add performance categories if score columns exist
                    score_cols = [col for col in df.columns if 'math' in col or 'science' in col or 'english' in col]
                    if score_cols:
                        df['avg_score'] = df[score_cols].mean(axis=1)
                        df['performance_category'] = df['avg_score'].apply(categorize_performance)
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.uploaded_file_name = current_file_name
                    st.session_state.uploaded_file_name_state = current_file_name
                    st.session_state.df_clean = df.copy()
                    
                    # Reset ML states
                    st.session_state.model_performance = None
                    st.session_state.performance_features = []
                    st.session_state.label_encoders = {}
                    st.session_state.best_model_name = "N/A"
                    st.session_state.accuracy = None
                    st.session_state.class_report = None
                
                st.success("✅ Data loaded and prepared successfully!")
                st.rerun()
            
            df = st.session_state.df
            
            # Show data overview
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📋 Your Data at a Glance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total Students", df.shape[0])
            with col2:
                st.metric("📊 Data Points", df.shape[1])
            with col3:
                if 'performance_category' in df.columns:
                    excellent = len(df[df['performance_category'] == 'Excellent'])
                    st.metric("⭐ Excellent", excellent)
                else:
                    st.metric("⭐ Complete Records", df.shape[0])
            with col4:
                if 'attendance_percentage' in df.columns:
                    avg_att = df['attendance_percentage'].mean()
                    st.metric("📅 Avg Attendance", f"{avg_att:.1f}%")
                else:
                    st.metric("✅ Ready", "Yes")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔍 Preview Your Data")
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Make sure your file is a valid CSV with student data")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3em; opacity: 0.7;'>
                <h2>👆 Click above to upload your student data file</h2>
                <p>We'll automatically prepare it for analysis</p>
            </div>
        """, unsafe_allow_html=True)

# ===========================
# Data Insights (OLAP Operations)
# ===========================
elif selected == "📊 Data Insights":
    st.markdown("<div class='section-header'><h2>📊 OLAP Data Exploration</h2></div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload your student data first!")
        st.info("👉 Go to '📤 Upload Data' in the sidebar to get started")
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.lower()

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in ['student_id', 'name']]

        st.markdown("### 🧭 Choose OLAP Operation")
        operation = st.selectbox(
            "Select OLAP Operation:",
            ["Roll-Up", "Roll-Down", "Slice", "Dice"],
            help="Choose the OLAP operation to explore data"
        )

        st.markdown("---")

        if operation in ["Roll-Up", "Roll-Down"]:
            col1, col2 = st.columns(2)
            higher_level = col1.selectbox("Higher-Level Category (e.g., Class):", categorical_cols)
            lower_level = col2.selectbox("Lower-Level Category (e.g., Section):", [c for c in categorical_cols if c != higher_level])
            measure = st.selectbox("Numeric column to analyze (e.g., Marks, Attendance):", numeric_cols)

            if st.button("🔍 Perform Operation", use_container_width=True):
                with st.spinner("Analyzing your data..."):
                    if operation == "Roll-Up":
                        result = df.groupby(higher_level)[measure].mean().reset_index()
                        st.markdown(f"### 📈 Rolled-Up: Average {measure.title()} by {higher_level.title()}")
                        st.bar_chart(result.set_index(higher_level))
                        st.dataframe(result)
                        st.success(f"✅ Data summarized at higher level ({higher_level})")

                    elif operation == "Roll-Down":
                        result = df.groupby([higher_level, lower_level])[measure].mean().reset_index()
                        st.markdown(f"### 📉 Rolled-Down: {measure.title()} by {higher_level.title()} → {lower_level.title()}")
                        for value in df[higher_level].unique():
                            subset = result[result[higher_level] == value]
                            st.markdown(f"#### {higher_level.title()}: {value}")
                            st.bar_chart(subset.set_index(lower_level)[measure])
                        st.dataframe(result)
                        st.success(f"✅ Data expanded to lower level ({lower_level})")

        elif operation == "Slice":
            st.markdown("### 🍰 Slice Data by a Single Dimension")
            slice_col = st.selectbox("Select column to slice by:", categorical_cols)
            slice_value = st.selectbox("Select value to view:", df[slice_col].unique().tolist())
            measure = st.selectbox("Numeric column to view:", numeric_cols)

            if st.button("🔎 Apply Slice", use_container_width=True):
                sliced_df = df[df[slice_col] == slice_value]
                result = sliced_df.groupby(slice_col)[measure].mean().reset_index()
                st.markdown(f"### 📊 Sliced Data — {slice_col.title()} = {slice_value}")
                st.bar_chart(result.set_index(slice_col))
                st.dataframe(sliced_df)
                st.success(f"✅ Showing only {slice_col} = {slice_value}")

        elif operation == "Dice":
            st.markdown("### 🎲 Dice Data by Multiple Dimensions")
            col1, col2 = st.columns(2)
            dim1 = col1.selectbox("Dimension 1:", categorical_cols)
            val1 = col1.multiselect(f"Select values for {dim1}:", df[dim1].unique().tolist())

            dim2 = col2.selectbox("Dimension 2:", [c for c in categorical_cols if c != dim1])
            val2 = col2.multiselect(f"Select values for {dim2}:", df[dim2].unique().tolist())

            measure = st.selectbox("Numeric column to analyze:", numeric_cols)

            if st.button("🎯 Apply Dice", use_container_width=True):
                diced_df = df[(df[dim1].isin(val1)) & (df[dim2].isin(val2))]
                result = diced_df.groupby([dim1, dim2])[measure].mean().reset_index()
                st.markdown(f"### 📈 Diced Data — {dim1}, {dim2} on {measure.title()}")
                st.dataframe(result)
                st.bar_chart(result, x=dim1, y=measure, color=dim2)
                st.success(f"✅ Applied dice filter for {dim1} & {dim2}")


# ===========================
# Student Groups
# ===========================
elif selected == "👥 Student Groups":
    st.markdown("<div class='section-header'><h2>👥 Discover Student Groups</h2></div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload your student data first!")
    else:
        df = st.session_state.df.copy()

        st.markdown("""
            <div class='info-box'>
                <p><strong>What is this?</strong> We'll automatically find groups of students who are similar to each other. 
                This helps you understand different types of learners or trends (like attendance, height, or weight) in your class!</p>
            </div>
        """, unsafe_allow_html=True)

        df.columns = df.columns.str.lower()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            selected_features = st.multiselect(
                "📊 Select one or two features to group students by:",
                options=numeric_cols,
                help="Pick 1 or 2 numeric features (e.g., math scores, attendance, height)"
            )

        with col2:
            n_clusters = st.slider(
                "👥 How many groups to create:",
                min_value=2, max_value=5, value=3,
                help="More groups = more specific categories"
            )

        # Elbow method helper
        if st.checkbox("📈 Show me how to choose the right number of groups"):
            if len(selected_features) >= 1:
                with st.spinner("🔄 Calculating..."):
                    X = df[selected_features]
                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = imputer.fit_transform(X)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imputed)
                    inertia = []
                    k_range = range(1, 11)

                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
                        inertia.append(km.inertia_)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(k_range, inertia, marker='o', color='#f093fb', linewidth=3, markersize=10)
                    ax.set_title('Finding the Sweet Spot for Number of Groups', color="#FFFFFF", fontsize=16, fontweight='bold')
                    ax.set_xlabel('Number of Groups', color="#FFFFFF", fontsize=13)
                    ax.set_ylabel('How Tight the Groups Are', color="#FFFFFF", fontsize=13)
                    ax.set_facecolor("rgba(255, 255, 255, 0.05)")
                    fig.patch.set_facecolor("#00000000")
                    ax.tick_params(colors="#FFFFFF")
                    ax.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.info("💡 Look for the 'elbow' — where the line starts to flatten. That's usually the best number of groups!")

        st.markdown("---")
        if st.button("🔍 Create Student Groups", use_container_width=True):
            if len(selected_features) == 0:
                st.warning("⚠️ Please select at least one feature for grouping.")
            else:
                with st.spinner("🤖 AI is creating student groups..."):
                    df_clustered = perform_clustering(df, selected_features, n_clusters)

                    # ============== Dynamic Group Naming Logic ==============
                    def generate_labels(feature_name, n_clusters):
                        name = feature_name.lower()
                        if 'attend' in name:
                            return ['Low Attendance', 'Moderate Attendance', 'High Attendance'][:n_clusters]
                        elif 'height' in name:
                            return ['Short', 'Average Height', 'Tall'][:n_clusters]
                        elif 'weight' in name:
                            return ['Underweight', 'Normal Weight', 'Overweight'][:n_clusters]
                        elif any(sub in name for sub in ['math', 'science', 'english', 'marathi', 'social']):
                            return ['Weak Students', 'Average Students', 'Good Students'][:n_clusters]
                        else:
                            return ['Low', 'Medium', 'High'][:n_clusters]

                    if df_clustered is not None:
                        st.success(f"✅ Created {n_clusters} student groups successfully!")

                        main_feature = selected_features[0]
                        group_labels = generate_labels(main_feature, n_clusters)
                        label_map = {i: group_labels[i] if i < len(group_labels) else f'Group {i+1}' for i in range(n_clusters)}
                        df_clustered['Group Label'] = df_clustered['Group'].map(label_map)

                        # ========================= Visualization Section =========================
                        cluster_sizes = df_clustered['Group Label'].value_counts().sort_index()

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("### 📊 Group Sizes")
                            fig, ax = plt.subplots(figsize=(8, 8))
                            colors = plt.cm.Set3(np.arange(len(cluster_sizes)))
                            wedges, texts, autotexts = ax.pie(
                                cluster_sizes.values,
                                labels=cluster_sizes.index,
                                autopct='%1.1f%%',
                                colors=colors,
                                startangle=90,
                                textprops={'color': 'white', 'fontsize': 12, 'fontweight': 'bold'}
                            )
                            for autotext in autotexts:
                                autotext.set_color('black')
                                autotext.set_fontweight('bold')
                            ax.set_title("How Students Are Distributed", color="#FFFFFF", fontsize=14, fontweight='bold')
                            fig.patch.set_facecolor("#00000000")

                            # ✅ FIXED legend color (tuple format)
                            legend = ax.legend(
                                cluster_sizes.index,
                                title="Groups",
                                facecolor=(0, 0, 0, 0.4),
                                labelcolor="white",
                                fontsize=11,
                                edgecolor="white"
                            )
                            legend.get_frame().set_alpha(0.8)
                            legend.get_frame().set_edgecolor("white")
                            st.pyplot(fig)

                        with col2:
                            st.markdown("### 📈 Group Characteristics")
                            cluster_stats = df_clustered.groupby('Group Label')[selected_features].mean().round(2).reset_index()
                            st.dataframe(cluster_stats, use_container_width=True)

                        st.markdown("---")

                        # === Dynamic Visualization (1D or 2D) ===
                        if len(selected_features) == 1:
                            feat = selected_features[0]
                            st.markdown(f"### 🎨 Visualizing Groups by {feat.title()}")

                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(data=df_clustered, x=feat, hue="Group Label", multiple="stack", bins=15, palette="Set2", ax=ax)
                            ax.set_title(f"Group Distribution by {feat.title()}", color="#FFFFFF", fontsize=14, fontweight='bold')
                            ax.set_xlabel(feat.title(), color="#FFFFFF")
                            ax.set_ylabel("Number of Students", color="#FFFFFF")
                            ax.tick_params(colors="#FFFFFF")

                            legend = ax.legend(
                                title="Groups",
                                facecolor=(0, 0, 0, 0.5),
                                labelcolor="white",
                                fontsize=11,
                                edgecolor="white"
                            )
                            legend.get_frame().set_alpha(0.8)
                            legend.get_frame().set_edgecolor("white")

                            fig.patch.set_facecolor("#00000000")
                            st.pyplot(fig)

                            st.info(f"🧠 **Observation:** Students with higher {feat} values generally fall under '{group_labels[-1]}' group.")

                        elif len(selected_features) == 2:
                            st.markdown("### 🎨 Visual Map of Student Groups")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            for label in df_clustered['Group Label'].unique():
                                cluster_data = df_clustered[df_clustered['Group Label'] == label]
                                ax.scatter(
                                    cluster_data[selected_features[0]],
                                    cluster_data[selected_features[1]],
                                    label=label,
                                    s=150, alpha=0.7,
                                    edgecolors='white', linewidth=2
                                )
                            ax.set_title(f"Student Groups Based on {selected_features[0].title()} and {selected_features[1].title()}",
                                        color="#FFFFFF", fontsize=16, fontweight='bold', pad=20)
                            ax.set_xlabel(selected_features[0].title(), color="#FFFFFF", fontsize=13)
                            ax.set_ylabel(selected_features[1].title(), color="#FFFFFF", fontsize=13)
                            ax.tick_params(colors="#FFFFFF")
                            ax.grid(True, alpha=0.3, linestyle='--')
                            fig.patch.set_facecolor("#00000000")

                            # ✅ FIXED legend color (tuple format)
                            legend = ax.legend(
                                title="Groups",
                                facecolor=(0, 0, 0, 0.4),
                                labelcolor="white",
                                fontsize=11,
                                edgecolor="white"
                            )
                            legend.get_frame().set_alpha(0.8)
                            legend.get_frame().set_edgecolor("white")

                            st.pyplot(fig)
                            st.info(f"🧠 **Observation:** Students with higher values in both {selected_features[0]} and {selected_features[1]} are part of '{group_labels[-1]}' group.")

                        st.markdown("---")
                        st.markdown("### 📋 Detailed Group Profiles")

                        for label in df_clustered['Group Label'].unique():
                            cluster_data = df_clustered[df_clustered['Group Label'] == label]
                            with st.expander(f"👥 {label} - {len(cluster_data)} students", expanded=False):
                                col1_exp, col2_exp = st.columns(2)
                                with col1_exp:
                                    st.markdown("**📊 Average Performance:**")
                                    avg_vals = cluster_data[selected_features].mean().round(2)
                                    for feat, val in avg_vals.items():
                                        st.metric(feat.replace('_', ' ').title(), f"{val:.2f}")

                                with col2_exp:
                                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                                    if len(categorical_cols) > 0:
                                        st.markdown("**🏷️ Common Traits:**")
                                        for cat_col in [c for c in categorical_cols if c not in ['student_id', 'name']][:3]:
                                            if cat_col in df.columns and not cluster_data[cat_col].empty:
                                                top = cluster_data[cat_col].mode()
                                                st.write(f"**{cat_col.replace('_', ' ').title()}:** {top.iloc[0] if not top.empty else 'N/A'}")

                                st.markdown("**📝 Sample Students from this Group:**")
                                display_cols = [c for c in cluster_data.columns if c not in ['Group', 'avg_score']][:5]
                                st.dataframe(cluster_data[display_cols].head(3), use_container_width=True)

                        # ================= Download Option =================
                        csv = df_clustered.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download Grouped Data",
                            data=csv,
                            file_name=f"student_groups_{n_clusters}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


# ===========================
# Performance Prediction
# ===========================
elif selected == "🎯 Performance Prediction":
    st.markdown("<div class='section-header'><h2>🎯 Predict Student Success</h2></div>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload your student data first!")
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.lower()
        
        # Auto-create performance categories if needed
        if 'performance_category' not in df.columns:
            score_cols = [col for col in df.columns if 'math' in col or 'science' in col or 'english' in col]
            if score_cols:
                df['avg_score'] = df[score_cols].mean(axis=1)
                df['performance_category'] = df['avg_score'].apply(categorize_performance)
                st.session_state.df = df
        
        # Train model if not already trained
        if st.session_state.model_performance is None:
            with st.spinner("🤖 Our AI is learning from your data..."):
                train_performance_model(df, PREDICTION_INPUT_FEATURES)
                
                if st.session_state.model_performance is None:
                    st.error("❌ Unable to train the predictor. Please check your data.")
                    st.stop()
        
        # Show model info
        if st.session_state.accuracy is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 AI Accuracy", f"{st.session_state.accuracy*100:.1f}%")
            with col2:
                st.metric("🧠 Prediction Model", st.session_state.best_model_name)
            with col3:
                st.metric("📊 Data Points Used", len(df))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prediction Form
        st.markdown("### 🔮 Make a Prediction")
        st.markdown("""
            <div class='info-box'>
                <p>Fill in the student information below and our AI will predict their likely performance level.</p>
            </div>
        """, unsafe_allow_html=True)
        
        base_features = [f for f in PREDICTION_INPUT_FEATURES if f in df.columns]
        
        # User-friendly labels
        friendly_labels = {
            'gender': '👤 Gender', 'class': '🎓 Class/Grade', 'age': '📅 Age',
            'attendance_percentage': '📅 How often they attend (%)', 
            'weight': '⚖️ Weight (kg)', 'height': '📏 Height (cm)', 
            'health_score': '💪 Health Condition (1-5)',
            'math_prev3': '🔢 Math Score (3 months ago)', 
            'math_prev2': '🔢 Math Score (2 months ago)',
            'math_prev1': '🔢 Math Score (last month)', 
            'science_prev2': '🔬 Science Score (2 months ago)',
            'science_prev1': '🔬 Science Score (last month)', 
            'english_prev3': '📖 English Score (3 months ago)',
            'english_prev2': '📖 English Score (2 months ago)', 
            'english_prev1': '📖 English Score (last month)'
        }
        
        with st.form("prediction_form"):
            cols = st.columns(3)
            input_data = {}
            
            for i, feat in enumerate(base_features):
                if feat not in df.columns:
                    continue
                
                with cols[i % 3]:
                    label = friendly_labels.get(feat, feat.replace('_', ' ').title())
                    
                    if df[feat].dtype == 'object':
                        options = sorted(df[feat].dropna().unique().tolist())
                        default_val = df[feat].mode()[0] if not df[feat].mode().empty else options[0]
                        default_index = options.index(default_val) if default_val in options else 0
                        
                        input_data[feat] = st.selectbox(label, options, index=default_index, key=f"pred_{feat}")
                            
                    elif df[feat].dtype in ['int64', 'float64'] and not df[feat].dropna().empty:
                        min_val = float(df[feat].min())
                        max_val = float(df[feat].max())
                        default_val = float(df[feat].mean())
                        default_val = max(min_val, min(max_val, default_val))
                        
                        input_data[feat] = st.slider(label, min_val, max_val, default_val, key=f"pred_{feat}")

            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.form_submit_button("🔮 Predict Performance", use_container_width=True)
            
            if predict_button:
                with st.spinner("🤖 AI is analyzing..."):
                    # Prepare input
                    actual_input = {}
                    for feat in base_features:
                        if feat in input_data:
                            actual_input[feat] = input_data[feat]
                        elif feat in df.columns:
                            mode_val = df[feat].mode()[0] if df[feat].dtype == 'object' and not df[feat].mode().empty else (
                                df[feat].mean() if not df[feat].dropna().empty else 0)
                            actual_input[feat] = mode_val

                    # Get prediction
                    prediction_result = classify_performance(actual_input, return_probabilities=True)
                    
                    if prediction_result is not None:
                        predicted_performance, probabilities = prediction_result
                        
                        # Display result with styling
                        performance_emoji = {
                            'Excellent': '🌟',
                            'Good': '👍',
                            'Needs Improvement': '📚'
                        }
                        
                        emoji = performance_emoji.get(predicted_performance, '📊')
                        
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, rgba(255,215,0,0.2) 0%, rgba(255,107,107,0.2) 100%);
                                        padding: 2em; border-radius: 20px; text-align: center; margin: 20px 0;
                                        border: 2px solid rgba(255,215,0,0.5);'>
                                <h1 style='font-size: 3em; margin: 0;'>{emoji}</h1>
                                <h2 style='color: #FFD700; margin: 10px 0;'>Predicted Performance</h2>
                                <h1 style='font-size: 2.5em; color: #FFFFFF; margin: 0;'>{predicted_performance}</h1>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show confidence levels
                        st.markdown("### 📊 How confident is our prediction?")
                        
                        for category, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(prob)
                            with col2:
                                st.markdown(f"**{category}**: {prob*100:.1f}%")
                        
                        # Recommendations
                        st.markdown("---")
                        st.markdown("### 💡 What can help?")
                        
                        if predicted_performance == 'Needs Improvement':
                            st.markdown("""
                                <div class='insight-card'>
                                    <p>📚 <strong>Extra Practice:</strong> Regular study sessions can make a big difference</p>
                                    <p>👥 <strong>Study Group:</strong> Learning with friends helps understanding</p>
                                    <p>🎯 <strong>Focus Areas:</strong> Identify weak subjects and work on them</p>
                                    <p>⏰ <strong>Time Management:</strong> Create a study schedule</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif predicted_performance == 'Good':
                            st.markdown("""
                                <div class='insight-card'>
                                    <p>⭐ <strong>Keep Going:</strong> You're on the right track!</p>
                                    <p>📈 <strong>Challenge Yourself:</strong> Try harder problems to improve further</p>
                                    <p>🎓 <strong>Help Others:</strong> Teaching others strengthens your own knowledge</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:  # Excellent
                            st.markdown("""
                                <div class='insight-card'>
                                    <p>🌟 <strong>Outstanding!</strong> Keep up the excellent work!</p>
                                    <p>🚀 <strong>Leadership:</strong> Consider helping classmates who need support</p>
                                    <p>🎯 <strong>Advanced Topics:</strong> Explore beyond the curriculum</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Unable to make prediction. Please try again.")

# ===========================
# Visual Analysis
# ===========================
elif selected == "📈 Visual Analysis":
    st.markdown("<div class='section-header'><h2>📈 Visualize Your Data</h2></div>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload your student data first!")
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.lower()
        
        st.markdown("### 🎨 Choose how you want to see your data")
        
        viz_options = {
            "📊 Compare Groups": "bar",
            "📈 See Trends": "line",
            "🔍 Find Patterns": "scatter",
            "📉 View Distribution": "histogram"
        }
        
        selected_viz = st.selectbox("Pick a visualization style:", list(viz_options.keys()))
        viz_type = viz_options[selected_viz]
        
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        # BAR CHART
        if viz_type == "bar":
            st.markdown("""
                <div class='info-box'>
                    <p>📊 Compare different groups of students and see which performs best</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            x_col = col1.selectbox("📌 Group students by:", categorical_cols, key="bar_x")
            y_col = col2.selectbox("📊 What to measure:", numeric_cols, key="bar_y")
            agg_choice = col2.radio("How to calculate:", ["Average", "Total", "Count"], horizontal=True)
            
            if st.button("🎨 Show Chart", use_container_width=True):
                if x_col and y_col:
                    try:
                        agg_map = {'Average': 'mean', 'Total': 'sum', 'Count': 'count'}
                        agg_func = agg_map[agg_choice]
                        
                        if agg_func == 'count':
                            plot_df = df.groupby(x_col)[y_col].count().reset_index(name='Count')
                            plot_y = 'Count'
                        else:
                            plot_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index(name=f'{agg_choice} {y_col}')
                            plot_y = f'{agg_choice} {y_col}'

                        fig, ax = plt.subplots(figsize=(12, 7))
                        colors = plt.cm.Spectral(np.linspace(0, 1, len(plot_df)))
                        bars = ax.bar(plot_df[x_col], plot_df[plot_y], color=colors, edgecolor='white', linewidth=2)
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}', ha='center', va='bottom', 
                                   color='white', fontweight='bold', fontsize=11)
                        
                        ax.set_title(f"Comparing {y_col.replace('_', ' ').title()} across {x_col.title()}", 
                                   color="#FFFFFF", fontsize=18, fontweight='bold', pad=20)
                        
                        fig.patch.set_facecolor("#00000000")
                        ax.set_xlabel(x_col.replace('_', ' ').title(), color="#FFFFFF", fontsize=14, fontweight='bold')
                        ax.set_ylabel(plot_y, color="#FFFFFF", fontsize=14, fontweight='bold')
                        ax.tick_params(colors="#FFFFFF", axis='x', labelrotation=45, labelsize=12)
                        ax.tick_params(colors="#FFFFFF", axis='y', labelsize=12)
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Insights
                        st.markdown("---")
                        insights = generate_insights("Bar Chart", df, x_col, y_col, agg_func)
                        
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown('<div class="insight-title">💡 What This Tells Us</div>', unsafe_allow_html=True)
                        for insight in insights:
                            st.markdown(f"• {insight}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error creating chart: {str(e)}")

        # HISTOGRAM
        elif viz_type == "histogram":
            st.markdown("""
                <div class='info-box'>
                    <p>📉 See how data is spread out - find the most common values</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            hist_col = col1.selectbox("📊 What to analyze:", numeric_cols, key="hist_col")
            bins = col2.slider("🔢 Detail level (more = more detail):", 5, 50, 20)
            
            if st.button("🎨 Show Distribution", use_container_width=True):
                if hist_col:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    
                    # Create histogram with gradient colors
                    n, bins_edges, patches = ax.hist(df[hist_col].dropna(), bins=bins, edgecolor='white', linewidth=1.5)
                    
                    # Color gradient
                    cm = plt.cm.viridis
                    for i, patch in enumerate(patches):
                        patch.set_facecolor(cm(i / len(patches)))
                    
                    # Add mean line
                    mean_val = df[hist_col].mean()
                    ax.axvline(mean_val, color='#FFD700', linestyle='--', linewidth=3, label=f'Average: {mean_val:.1f}')
                    
                    ax.set_title(f"Distribution of {hist_col.replace('_', ' ').title()}", 
                               color="#FFFFFF", fontsize=18, fontweight='bold', pad=20)
                    
                    fig.patch.set_facecolor("#00000000")
                    ax.set_xlabel(hist_col.replace('_', ' ').title(), color="#FFFFFF", fontsize=14, fontweight='bold')
                    ax.set_ylabel("Number of Students", color="#FFFFFF", fontsize=14, fontweight='bold')
                    ax.tick_params(colors="#FFFFFF", labelsize=12)
                    ax.legend(edgecolor="white", labelcolor="#FFFFFF", fontsize=12)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Insights
                    st.markdown("---")
                    insights = generate_insights("Distribution", df, hist_col)
                    
                    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title">💡 What This Tells Us</div>', unsafe_allow_html=True)
                    for insight in insights:
                        st.markdown(f"• {insight}")
                    st.markdown('</div>', unsafe_allow_html=True)

        # SCATTER PLOT
        elif viz_type == "scatter":
            st.markdown("""
                <div class='info-box'>
                    <p>🔍 See if two things are related - like attendance and grades</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("📊 First thing to compare:", numeric_cols, key="scatter_x")
            y_col = col2.selectbox("📊 Second thing to compare:", [c for c in numeric_cols if c != x_col], key="scatter_y")
            
            if st.button("🎨 Show Pattern", use_container_width=True):
                if x_col and y_col:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    
                    scatter = ax.scatter(df[x_col], df[y_col], 
                                       c=df[y_col], cmap='Spectral',
                                       s=150, alpha=0.6, edgecolors='white', linewidth=2)
                    
                    # Add trend line
                    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                    p = np.poly1d(z)
                    ax.plot(df[x_col], p(df[x_col]), "r--", linewidth=3, alpha=0.8, label='Trend')
                    
                    ax.set_title(f"Relationship: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}", 
                               color="#FFFFFF", fontsize=18, fontweight='bold', pad=20)
                    
                    fig.patch.set_facecolor("#00000000")
                    ax.set_xlabel(x_col.replace('_', ' ').title(), color="#FFFFFF", fontsize=14, fontweight='bold')
                    ax.set_ylabel(y_col.replace('_', ' ').title(), color="#FFFFFF", fontsize=14, fontweight='bold')
                    ax.tick_params(colors="#FFFFFF", labelsize=12)
                    ax.legend(edgecolor="white", labelcolor="#FFFFFF", fontsize=12)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(y_col.replace('_', ' ').title(), color='#FFFFFF', fontsize=12)
                    cbar.ax.tick_params(colors='#FFFFFF')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Insights
                    st.markdown("---")
                    insights = generate_insights("Scatter Plot", df, x_col, y_col)
                    
                    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title">💡 What This Tells Us</div>', unsafe_allow_html=True)
                    for insight in insights:
                        st.markdown(f"• {insight}")
                    st.markdown('</div>', unsafe_allow_html=True)

        # LINE CHART
        else:  # line
            st.markdown("""
                <div class='info-box'>
                    <p>📈 Track how things change - see progress over time or across categories</p>
                </div>
            """, unsafe_allow_html=True)
            
            selected_cols = st.multiselect("📊 Select things to track (pick 2-4):", 
                                          numeric_cols, 
                                          default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
            
            if st.button("🎨 Show Trends", use_container_width=True):
                if len(selected_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    
                    colors = plt.cm.Set3(np.linspace(0, 1, len(selected_cols)))
                    
                    for i, col in enumerate(selected_cols):
                        ax.plot(df.index, df[col], label=col.replace('_', ' ').title(), 
                               linewidth=3, marker='o', markersize=6, color=colors[i])
                    
                    ax.set_title("Trend Comparison", 
                               color="#FFFFFF", fontsize=18, fontweight='bold', pad=20)
                    
                    fig.patch.set_facecolor("#00000000")
                    ax.set_xlabel("Student Record", color="#FFFFFF", fontsize=14, fontweight='bold')
                    ax.set_ylabel("Values", color="#FFFFFF", fontsize=14, fontweight='bold')
                    ax.tick_params(colors="#FFFFFF", labelsize=12)
                    ax.legend(facecolor="rgba(255, 255, 255, 0.1)", edgecolor="white", 
                             labelcolor="#FFFFFF", fontsize=11, loc='best')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Summary stats
                    st.markdown("---")
                    st.markdown("### 📊 Quick Summary")
                    
                    summary_cols = st.columns(len(selected_cols))
                    for i, col in enumerate(selected_cols):
                        with summary_cols[i]:
                            avg = df[col].mean()
                            st.metric(col.replace('_', ' ').title(), f"{avg:.1f}", 
                                    delta=f"Range: {df[col].min():.0f}-{df[col].max():.0f}")
                else:
                    st.warning("⚠️ Please select at least 2 things to compare")

# ===========================
# Pattern Discovery (Teacher-Friendly Insights)
# ===========================
elif selected == "🔍 Pattern Discovery":
    st.markdown("<div class='section-header'><h2>🔍 Student Performance Patterns</h2></div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload your student data first!")
    else:
        df = st.session_state.df.copy()

        st.markdown("""
        <div class='info-box'>
            <p><strong>What this does:</strong> This tool automatically finds key learning patterns — 
            such as how attendance and gender affect performance — and presents them in easy language for teachers.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔎 Analyze Patterns", use_container_width=True):
            with st.spinner("🔍 Finding important patterns..."):
                rules, frequent_itemsets = discover_patterns(df)

                if rules is not None and len(rules) > 0:
                    # Sort and select top patterns
                    top_rules = rules.sort_values(["lift", "confidence"], ascending=False).head(10)

                    # Identify clear relationships
                    attendance_avg = df["attendance_percentage"].mean()

                    # Compute average marks (using available columns)
                    mark_cols = [col for col in df.columns if "math" in col or "science" in col or "english" in col]
                    avg_score = df[mark_cols].mean().mean() if mark_cols else 0

                    # Attendance vs Performance
                    high_attendance = df[df["attendance_percentage"] >= attendance_avg]
                    low_attendance = df[df["attendance_percentage"] < attendance_avg]

                    high_perf = high_attendance[mark_cols].mean().mean() if len(high_attendance) > 0 else 0
                    low_perf = low_attendance[mark_cols].mean().mean() if len(low_attendance) > 0 else 0

                    attendance_insight = (
                        "📘 Students with **high attendance** generally show **excellent performance**."
                        if high_perf > low_perf + 10
                        else "📗 Students with **good attendance** show **average to above-average performance**."
                        if high_perf > low_perf + 5
                        else "📙 Attendance shows **minor impact** on performance in this dataset."
                    )

                    # Gender vs Performance
                    if "gender" in df.columns:
                        male_avg = df[df["gender"].str.lower().isin(["male", "m"])]
                        female_avg = df[df["gender"].str.lower().isin(["female", "f"])]

                        male_perf = male_avg[mark_cols].mean().mean() if len(male_avg) > 0 else 0
                        female_perf = female_avg[mark_cols].mean().mean() if len(female_avg) > 0 else 0

                        if female_perf > male_perf + 5:
                            gender_insight = "👩 **Female students** tend to perform better overall than male students."
                        elif male_perf > female_perf + 5:
                            gender_insight = "👦 **Male students** tend to perform better overall than female students."
                        else:
                            gender_insight = "⚖️ Both genders perform nearly equally across most subjects."
                    else:
                        gender_insight = "⚠️ Gender data not available to analyze this pattern."

                    # Display Insights
                    st.success("✅ Insights Generated Successfully!")
                    st.markdown("### 🧠 Key Findings")
                    st.markdown(f"""
                        <div style='background:#1E1E1E;padding:15px;border-radius:10px;margin-bottom:10px;'>
                            <h4>1️⃣ Attendance vs Performance</h4>
                            <p style='font-size:1.1em'>{attendance_insight}</p>
                        </div>
                        <div style='background:#1E1E1E;padding:15px;border-radius:10px;margin-bottom:10px;'>
                            <h4>2️⃣ Gender-Based Performance</h4>
                            <p style='font-size:1.1em'>{gender_insight}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Visualization
                    st.markdown("---")
                    st.markdown("### 📊 Attendance vs Performance Visualization")

                    # Pick one math column automatically
                    math_col = next((col for col in df.columns if "math" in col.lower()), None)

                    if math_col:
                        plt.figure(figsize=(8, 5))
                        sns.scatterplot(
                            data=df,
                            x="attendance_percentage",
                            y=math_col,
                            hue="gender" if "gender" in df.columns else None,
                            alpha=0.7
                        )
                        plt.title(f"Attendance vs {math_col.replace('_', ' ').title()}", fontsize=14)
                        plt.xlabel("Attendance (%)")
                        plt.ylabel(f"{math_col.replace('_', ' ').title()}")
                        st.pyplot(plt)
                    else:
                        st.warning("⚠️ No Math-related column found for visualization.")

                    st.markdown("""
                    **Observation:**  
                    🔹 Students with attendance above average usually perform well.  
                    🔹 Attendance below 70% often results in weaker scores.  
                    🔹 Gender differences may vary across subjects.
                    """)
                else:
                    st.warning("⚠️ Not enough patterns found. Ensure your dataset contains attendance and subject performance columns.")


# ===========================
# AI Assistant
# ===========================
elif selected == "💬 AI Assistant":
    st.markdown("<div class='section-header'><h2>💬 Your Smart Education Helper</h2></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <p><strong>Ask me anything!</strong> I can help you understand student performance, 
            suggest teaching strategies, explain patterns in your data, or answer any education-related questions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if client is None:
        st.error("❌ AI Assistant is not available. Please contact your administrator.")
        st.info("💡 Admin: Check your GEMINI_API_KEY in secrets.toml")
    else:
        # Suggested questions
        st.markdown("### 💭 Try asking:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💡 How can I improve low attendance?", use_container_width=True):
                st.session_state.suggested_question = "How can I improve student attendance in my class?"
        
        with col2:
            if st.button("📚 What helps struggling students?", use_container_width=True):
                st.session_state.suggested_question = "What are effective strategies to help struggling students improve?"
        
        with col3:
            if st.button("🎯 How to boost engagement?", use_container_width=True):
                st.session_state.suggested_question = "How can I make my classes more engaging for students?"
        
        st.markdown("---")
        
        # Initialize chat
        SYSTEM_INSTRUCTION = (
            "You are a friendly, supportive AI assistant helping teachers and parents improve student education. "
            "Provide practical, easy-to-understand advice. Use simple language, be encouraging, and focus on actionable tips. "
            "When discussing data or patterns, explain them in plain terms that anyone can understand. "
            "Keep responses concise but helpful, around 3-5 short paragraphs unless asked for more detail."
        )
        
        if "gemini_chat" not in st.session_state:
            config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            st.session_state.gemini_chat = client.chats.create(
                model=GEMINI_MODEL, history=[], config=config)
            st.session_state.messages = []
        
        # Display chat history with custom styling
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); 
                                padding: 15px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #667eea;'>
                        <strong>👤 You:</strong><br>{msg["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%); 
                                padding: 15px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #f093fb;'>
                        <strong>🤖 AI Helper:</strong><br>{msg["content"]}
                    </div>
                """, unsafe_allow_html=True)
        
        # Handle suggested questions
        if 'suggested_question' in st.session_state:
            prompt = st.session_state.suggested_question
            del st.session_state.suggested_question
        else:
            prompt = st.chat_input("💬 Type your question here...")
        
        if prompt:
            current_lang = st.session_state.target_language_code
            
            # Translate to English if needed
            prompt_en = prompt
            if current_lang != 'en':
                try:
                    translator = Translator()
                    prompt_en = translator.translate(prompt, src=current_lang, dest='en').text
                except Exception:
                    prompt_en = prompt
            
            # Display user message
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); 
                            padding: 15px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #667eea;'>
                    <strong>👤 You:</strong><br>{prompt}
                </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            try:
                with st.spinner("🤔 Thinking..."):
                    # Add context about the current data if available
                    context_prompt = prompt_en
                    if st.session_state.df is not None:
                        df = st.session_state.df
                        num_students = len(df)
                        
                        # Get some basic stats
                        if 'attendance_percentage' in df.columns:
                            avg_attendance = df['attendance_percentage'].mean()
                            context_prompt = f"Context: I have data on {num_students} students with average attendance of {avg_attendance:.1f}%. Question: {prompt_en}"
                    
                    response = st.session_state.gemini_chat.send_message(context_prompt)
                    resp_text_en = response.text
                
                # Translate response back
                full_resp = resp_text_en
                if current_lang != 'en':
                    try:
                        translator = Translator()
                        full_resp = translator.translate(resp_text_en, dest=current_lang).text
                    except Exception:
                        full_resp = resp_text_en
                
                # Display assistant response
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%); 
                                padding: 15px; border-radius: 15px; margin: 10px 0; border-left: 4px solid #f093fb;'>
                        <strong>🤖 AI Helper:</strong><br>{full_resp}
                    </div>
                """, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": full_resp})
                
                # Auto-scroll to bottom
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Sorry, I couldn't process that. Please try again!")
                st.caption(f"Technical details: {str(e)}")
        
        # Clear chat button
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Start Fresh", use_container_width=True):
                config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
                st.session_state.gemini_chat = client.chats.create(
                    model=GEMINI_MODEL, history=[], config=config)
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            st.markdown("""
                <div style='padding: 10px; opacity: 0.7; font-size: 0.9em;'>
                    💡 <strong>Tip:</strong> Be specific in your questions for better answers!
                </div>
            """, unsafe_allow_html=True)