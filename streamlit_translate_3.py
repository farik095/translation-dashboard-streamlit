import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Translation Framework Analysis Dashboard",
    page_icon="🌍",
    layout="wide"
)

@st.cache_data
def load_data_from_file(file_path):
    """Load and preprocess data from file path"""
    try:
        df = pd.read_csv(file_path)
        return preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data from file: {e}")
        return None

@st.cache_data
def load_data_from_upload(uploaded_file):
    """Load and preprocess data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading uploaded data: {e}")
        return None

def preprocess_data(df):
    """Common data preprocessing function"""
    try:
        # Clean column names - remove trailing spaces and commas
        df.columns = df.columns.str.strip().str.rstrip(',')
        
        # Handle missing values
        df['From'] = df['From'].fillna('Unknown')
        df['To'] = df['To'].fillna('Unknown')
        df['Status'] = df['Status'].fillna('Unknown')
        
        # Convert boolean columns
        if 'Completed' in df.columns:
            df['Completed'] = df['Completed'].map({'Yes': True, 'No': False}).fillna(False)
        
        if 'Timed Out' in df.columns:
            df['Timed Out'] = df['Timed Out'].map({'Yes': True, 'No': False}).fillna(False)
        
        # Parse timestamp
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Hour'] = df['Timestamp'].dt.hour
            df['Date'] = df['Timestamp'].dt.date
        
        # Clean translation scores
        if 'Translation Score' in df.columns:
            df['Translation Score'] = pd.to_numeric(df['Translation Score'], errors='coerce')
        
        # Create translation direction
        df['Translation Direction'] = df['From'].astype(str) + ' → ' + df['To'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def create_summary_stats(df):
    """Calculate summary statistics"""
    total_translations = len(df)
    completed_translations = df['Completed'].sum() if 'Completed' in df.columns else 0
    timed_out_translations = df['Timed Out'].sum() if 'Timed Out' in df.columns else 0
    
    completion_rate = (completed_translations / total_translations * 100) if total_translations > 0 else 0
    timeout_rate = (timed_out_translations / total_translations * 100) if total_translations > 0 else 0
    
    avg_score = df['Translation Score'].mean() if 'Translation Score' in df.columns else 0
    
    return {
        'total_translations': total_translations,
        'completed_translations': completed_translations,
        'timed_out_translations': timed_out_translations,
        'completion_rate': completion_rate,
        'timeout_rate': timeout_rate,
        'avg_score': avg_score
    }

def create_direction_analysis(df):
    """Analyze results by translation direction"""
    direction_stats = []
    
    for direction, group in df.groupby('Translation Direction'):
        total = len(group)
        completed = group['Completed'].sum() if 'Completed' in group.columns else 0
        timed_out = group['Timed Out'].sum() if 'Timed Out' in group.columns else 0
        avg_score = group['Translation Score'].mean() if 'Translation Score' in group.columns else 0
        
        direction_stats.append({
            'Translation Direction': direction,
            'Total': total,
            'Completed': completed,
            'Timed Out': timed_out,
            'Completion Rate (%)': round(completed / total * 100, 1) if total > 0 else 0,
            'Timeout Rate (%)': round(timed_out / total * 100, 1) if total > 0 else 0,
            'Avg Score': round(avg_score, 2) if not pd.isna(avg_score) else 0
        })
    
    return pd.DataFrame(direction_stats)

def main():
    st.title("🌍 Translation Framework Analysis Dashboard")
    
    # Sidebar for file selection
    st.sidebar.header("Data Source")
    
    # File upload option
    st.sidebar.subheader("📁 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # Default file option
    st.sidebar.subheader("📂 Or Use Default File")
    use_default_file = st.sidebar.checkbox("Use default file", value=True)
    
    df = None
    
    if uploaded_file is not None and not use_default_file:
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        df = load_data_from_upload(uploaded_file)
        
    elif use_default_file:
        default_file_path = "Translation Framework Test Results @ 08.19.csv"
        
        if os.path.exists(default_file_path):
            st.sidebar.success(f"✅ Using default file")
            df = load_data_from_file(default_file_path)
        else:
            st.sidebar.error(f"❌ Default file not found: {default_file_path}")
            st.error(f"Default file not found: {default_file_path}")
    else:
        st.info("👆 Please upload a CSV file or check 'Use default file' in the sidebar.")
        return
    
    if df is None or len(df) == 0:
        st.warning("No valid data loaded. Please check your file.")
        return
    
    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Info")
    st.sidebar.info(f"**Rows:** {len(df)}\n**Columns:** {len(df.columns)}")
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Date filter
    if 'Timestamp' in df.columns and not df['Timestamp'].isna().all():
        min_date = df['Timestamp'].min().date()
        max_date = df['Timestamp'].max().date()
        
        st.sidebar.markdown("📅 **Date Range**")
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_filter"
        )
        
        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[
                (df['Timestamp'].dt.date >= start_date) & 
                (df['Timestamp'].dt.date <= end_date)
            ]
        elif len(date_range) == 1:
            selected_date = date_range[0]
            df = df[df['Timestamp'].dt.date == selected_date]
    
    # Translation direction filter
    unique_directions = df['Translation Direction'].unique()
    directions = ['All'] + sorted(unique_directions.tolist())
    selected_direction = st.sidebar.selectbox("Translation Direction", directions)
    
    if selected_direction != 'All':
        df = df[df['Translation Direction'] == selected_direction]
    
    # Summary statistics
    stats = create_summary_stats(df)
    
    st.subheader("📊 Overall Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Translations", stats['total_translations'])
    
    with col2:
        st.metric("Completion Rate", f"{stats['completion_rate']:.1f}%")
    
    with col3:
        st.metric("Timeout Rate", f"{stats['timeout_rate']:.1f}%")
    
    with col4:
        st.metric("Avg Score", f"{stats['avg_score']:.2f}")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔄 By Direction", "📋 Raw Data"])
    
    with tab1:
        st.subheader("Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple pie chart
            labels = ['Completed', 'Timed Out', 'Other']
            values = [
                stats['completed_translations'], 
                stats['timed_out_translations'], 
                stats['total_translations'] - stats['completed_translations'] - stats['timed_out_translations']
            ]
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig_pie.update_layout(title="Translation Results")
            st.plotly_chart(fig_pie, use_container_width=True, key="overview_pie")
        
        with col2:
            # Score histogram
            if 'Translation Score' in df.columns:
                scores = df['Translation Score'].dropna()
                if len(scores) > 0:
                    fig_hist = go.Figure(data=[go.Histogram(x=scores)])
                    fig_hist.update_layout(title="Score Distribution", xaxis_title="Score", yaxis_title="Count")
                    st.plotly_chart(fig_hist, use_container_width=True, key="overview_hist")
    
    with tab2:
        st.subheader("Performance by Translation Direction")
        direction_stats = create_direction_analysis(df)
        
        if not direction_stats.empty:
            st.dataframe(direction_stats, use_container_width=True)
            
            # Bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Completed',
                x=direction_stats['Translation Direction'],
                y=direction_stats['Completed']
            ))
            fig_bar.add_trace(go.Bar(
                name='Timed Out',
                x=direction_stats['Translation Direction'],
                y=direction_stats['Timed Out']
            ))
            
            fig_bar.update_layout(
                title="Results by Direction",
                xaxis_title="Translation Direction",
                yaxis_title="Count",
                barmode='stack'
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="direction_bar")
    
    with tab3:
        st.subheader("Raw Data")
        
        # Show basic columns by default
        basic_columns = ['Index', 'Original Text', 'From', 'To', 'AI Response', 'Status', 'Translation Score']
        display_columns = [col for col in basic_columns if col in df.columns]
        
        st.dataframe(df[display_columns], use_container_width=True)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv_data,
            file_name=f"translation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()