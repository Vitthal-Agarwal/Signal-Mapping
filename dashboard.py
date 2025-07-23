"""
Real-time Behavioral Signal Monitoring Dashboard
Interactive dashboard for monitoring disengagement patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from multi_agent_system import MultiAgentDisengagementDetector
except ImportError:
    st.error("Could not import multi_agent_system. Please ensure the file exists in the same directory.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Behavioral Signal Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.high-risk {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}
.medium-risk {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
}
.low-risk {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        data = pd.read_csv("synthetic_user_behavior_log.csv")
        return data
    except FileNotFoundError:
        st.error("Could not find synthetic_user_behavior_log.csv. Please ensure the file exists.")
        return pd.DataFrame()

@st.cache_data
def analyze_with_agents(data):
    """Run multi-agent analysis and cache results"""
    if data.empty:
        return pd.DataFrame(), {}
    
    detector = MultiAgentDisengagementDetector()
    results = detector.analyze_dataset(data)
    summary = detector.get_summary_report()
    
    return results, summary

def create_risk_distribution_chart(results):
    """Create risk score distribution chart"""
    fig = px.histogram(
        results, 
        x='risk_score', 
        nbins=20,
        title='Risk Score Distribution',
        labels={'risk_score': 'Risk Score', 'count': 'Number of Users'},
        color_discrete_sequence=['#3366cc']
    )
    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Number of Users",
        showlegend=False
    )
    return fig

def create_alert_level_chart(results):
    """Create alert level distribution chart"""
    alert_counts = results['highest_alert_level'].value_counts()
    
    colors = {
        'LOW': '#4caf50',
        'MEDIUM': '#ff9800', 
        'HIGH': '#f44336',
        'CRITICAL': '#9c27b0'
    }
    
    fig = px.pie(
        values=alert_counts.values,
        names=alert_counts.index,
        title='Alert Level Distribution',
        color=alert_counts.index,
        color_discrete_map=colors
    )
    return fig

def create_risk_timeline_chart(results):
    """Create simulated risk timeline"""
    # Simulate historical data for demonstration
    dates = pd.date_range(start='2025-01-01', end='2025-07-23', freq='D')
    np.random.seed(42)
    
    # Create simulated metrics over time
    timeline_data = []
    for date in dates:
        # Simulate increasing disengagement over time with some noise
        base_risk = 0.2 + 0.3 * (date - dates[0]).days / len(dates)
        daily_risk = max(0, min(1, base_risk + np.random.normal(0, 0.1)))
        
        timeline_data.append({
            'date': date,
            'avg_risk_score': daily_risk,
            'high_risk_users': int(len(results) * daily_risk * np.random.uniform(0.1, 0.3)),
            'total_alerts': int(len(results) * daily_risk * np.random.uniform(0.2, 0.5))
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Risk Score Over Time', 'Alert Counts Over Time'),
        vertical_spacing=0.1
    )
    
    # Risk score line
    fig.add_trace(
        go.Scatter(
            x=timeline_df['date'],
            y=timeline_df['avg_risk_score'],
            name='Avg Risk Score',
            line=dict(color='#f44336', width=2)
        ),
        row=1, col=1
    )
    
    # Alert counts
    fig.add_trace(
        go.Scatter(
            x=timeline_df['date'],
            y=timeline_df['total_alerts'],
            name='Total Alerts',
            fill='tonexty',
            line=dict(color='#ff9800', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Risk Trends Over Time"
    )
    
    return fig

def create_user_detail_view(results, user_id):
    """Create detailed view for a specific user"""
    user_data = results[results['user_id'] == user_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Risk Score",
            value=f"{user_data['risk_score']:.3f}",
            delta=f"Alert Level: {user_data['highest_alert_level']}"
        )
    
    with col2:
        st.metric(
            label="Alert Count",
            value=user_data['alert_count']
        )
    
    with col3:
        risk_level = user_data['highest_alert_level']
        if risk_level == 'CRITICAL':
            st.error(f"🚨 {risk_level} RISK")
        elif risk_level == 'HIGH':
            st.warning(f"⚠️ {risk_level} RISK")
        elif risk_level == 'MEDIUM':
            st.info(f"ℹ️ {risk_level} RISK")
        else:
            st.success(f"✅ {risk_level} RISK")

def main():
    """Main dashboard function"""
    st.title("🎯 Behavioral Signal Monitoring Dashboard")
    st.markdown("Real-time monitoring of team disengagement patterns using multi-agent analysis")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Load data
    data = load_data()
    
    if data.empty:
        st.error("No data available. Please check your data file.")
        return
    
    # Run analysis
    with st.spinner("Running multi-agent analysis..."):
        results, summary = analyze_with_agents(data)
    
    if results.empty:
        st.error("Analysis failed. Please check your data and try again.")
        return
    
    # Main metrics
    st.header("📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Users",
            value=len(results)
        )
    
    with col2:
        high_risk_count = len(results[results['risk_score'] > 0.6])
        st.metric(
            label="High Risk Users",
            value=high_risk_count,
            delta=f"{high_risk_count/len(results):.1%} of total"
        )
    
    with col3:
        avg_risk = results['risk_score'].mean()
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.3f}"
        )
    
    with col4:
        total_alerts = summary.get('total_alerts', 0)
        st.metric(
            label="Active Alerts",
            value=total_alerts
        )
    
    # Charts section
    st.header("📈 Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Alert Levels", "Trends"])
    
    with tab1:
        st.plotly_chart(create_risk_distribution_chart(results), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_alert_level_chart(results), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_risk_timeline_chart(results), use_container_width=True)
    
    # High-risk users table
    st.header("🚨 High-Risk Users")
    
    high_risk_users = results[results['risk_score'] > 0.6].sort_values('risk_score', ascending=False)
    
    if len(high_risk_users) > 0:
        for idx, user in high_risk_users.iterrows():
            risk_class = "high-risk" if user['risk_score'] > 0.8 else "medium-risk"
            
            st.markdown(f"""
            <div class="metric-container {risk_class}">
                <h4>{user['user_id']}</h4>
                <p><strong>Risk Score:</strong> {user['risk_score']:.3f}</p>
                <p><strong>Alert Level:</strong> {user['highest_alert_level']}</p>
                <p><strong>Alerts:</strong> {user['alert_count']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No high-risk users identified.")
    
    # User detail section
    st.header("👤 User Detail View")
    
    selected_user = st.selectbox(
        "Select a user to view details:",
        options=results['user_id'].tolist(),
        index=0
    )
    
    if selected_user:
        create_user_detail_view(results, selected_user)
    
    # Data export section
    st.header("💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Copy Results to Clipboard"):
            results_csv = results.to_csv(index=False)
            st.code(results_csv)
    
    with col2:
        csv_data = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"disengagement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Summary report
    st.header("📋 Summary Report")
    
    if summary:
        st.json(summary)
    
    # Auto-refresh option
    st.sidebar.header("⚙️ Settings")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    
    if auto_refresh:
        st.sidebar.success("Auto-refresh enabled")
        # Use st.rerun() for Streamlit >= 1.18.0, or st.experimental_rerun() for older versions
        try:
            import time
            time.sleep(30)
            st.rerun()
        except:
            st.sidebar.info("Manual refresh required")
    
    # Sidebar information
    st.sidebar.header("ℹ️ System Information")
    st.sidebar.info(f"""
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    **Analysis Method:** Multi-Agent System
    - Message Edit Agent
    - Meeting Pattern Agent  
    - Calendar Void Agent
    - Engagement Trend Agent
    
    **Risk Levels:**
    - LOW: 0.0 - 0.4
    - MEDIUM: 0.4 - 0.6
    - HIGH: 0.6 - 0.8
    - CRITICAL: 0.8 - 1.0
    """)

if __name__ == "__main__":
    main()
