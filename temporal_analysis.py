"""
Temporal Analysis and Trend Detection System
Advanced time-series analysis for behavioral pattern detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TemporalAnalyzer:
    """Advanced temporal analysis for behavioral patterns"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.trend_models = {}
        self.anomaly_scores = {}
        self.seasonal_patterns = {}
        
    def create_time_series(self, data: pd.DataFrame, time_window_days: int = 7) -> pd.DataFrame:
        """Create time series data from behavioral events"""
        
        print(f"Creating time series with {time_window_days}-day windows...")
        
        # Define time range
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 7, 23)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        time_series_data = []
        
        for user_idx, user_data in data.iterrows():
            user_id = user_data.get('user_id', f'User_{user_idx}')
            
            # Parse timestamps
            edit_times = self._parse_timestamps(user_data.get('message_edit_timestamps', ''))
            meeting_times = self._parse_timestamps(user_data.get('meeting_timestamps', ''))
            
            # Create daily metrics
            for date in date_range:
                window_start = date - timedelta(days=time_window_days)
                window_end = date
                
                # Count events in window
                edit_count = sum(1 for t in edit_times if window_start <= t <= window_end)
                meeting_count = sum(1 for t in meeting_times if window_start <= t <= window_end)
                
                # Calculate patterns
                edit_pattern = self._calculate_daily_pattern(edit_times, date, time_window_days)
                meeting_pattern = self._calculate_daily_pattern(meeting_times, date, time_window_days)
                
                # Calculate engagement metrics
                total_activity = edit_count + meeting_count
                engagement_ratio = total_activity / time_window_days if time_window_days > 0 else 0
                
                # Calculate velocity (change from previous period)
                prev_date = date - timedelta(days=time_window_days)
                prev_edit_count = sum(1 for t in edit_times if 
                                    prev_date - timedelta(days=time_window_days) <= t <= prev_date)
                prev_meeting_count = sum(1 for t in meeting_times if 
                                       prev_date - timedelta(days=time_window_days) <= t <= prev_date)
                
                edit_velocity = edit_count - prev_edit_count
                meeting_velocity = meeting_count - prev_meeting_count
                
                time_series_data.append({
                    'user_id': user_id,
                    'date': date,
                    'edit_count': edit_count,
                    'meeting_count': meeting_count,
                    'total_activity': total_activity,
                    'engagement_ratio': engagement_ratio,
                    'edit_velocity': edit_velocity,
                    'meeting_velocity': meeting_velocity,
                    'edit_pattern_score': edit_pattern,
                    'meeting_pattern_score': meeting_pattern,
                    'day_of_week': date.weekday(),
                    'week_of_year': date.isocalendar()[1],
                    'is_weekend': date.weekday() >= 5
                })
        
        ts_df = pd.DataFrame(time_series_data)
        print(f"Created time series: {len(ts_df)} records for {data.shape[0]} users over {len(date_range)} days")
        
        return ts_df
    
    def _parse_timestamps(self, timestamps_str: str) -> list:
        """Parse timestamp string into datetime objects"""
        if pd.isna(timestamps_str) or str(timestamps_str).strip() == '':
            return []
        
        try:
            times = [datetime.fromisoformat(t.strip()) for t in str(timestamps_str).split(",")]
            return times
        except:
            return []
    
    def _calculate_daily_pattern(self, timestamps: list, target_date: datetime, window_days: int) -> float:
        """Calculate pattern score for activity around a target date"""
        if not timestamps:
            return 0.0
        
        # Count activities by day within window
        window_start = target_date - timedelta(days=window_days)
        window_activities = [t for t in timestamps if window_start <= t <= target_date]
        
        if len(window_activities) < 2:
            return 0.0
        
        # Calculate regularity score based on gap consistency
        gaps = [(window_activities[i] - window_activities[i-1]).days 
                for i in range(1, len(window_activities))]
        
        if len(gaps) == 0:
            return 0.0
        
        # Lower standard deviation = more regular pattern
        gap_std = np.std(gaps)
        regularity_score = 1 / (1 + gap_std)
        
        return regularity_score
    
    def detect_trends(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """Detect trends in user engagement over time"""
        
        print("Detecting engagement trends...")
        
        trend_results = []
        
        for user_id in ts_df['user_id'].unique():
            user_ts = ts_df[ts_df['user_id'] == user_id].sort_values('date')
            
            if len(user_ts) < 10:  # Need minimum data points
                continue
            
            # Calculate trends for different metrics
            metrics = ['edit_count', 'meeting_count', 'total_activity', 'engagement_ratio']
            
            trends = {}
            for metric in metrics:
                values = user_ts[metric].values
                
                # Linear trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trends[f'{metric}_slope'] = slope
                trends[f'{metric}_correlation'] = r_value
                trends[f'{metric}_p_value'] = p_value
                
                # Recent vs historical comparison
                split_point = len(values) // 2
                recent_mean = np.mean(values[split_point:])
                historical_mean = np.mean(values[:split_point])
                
                trends[f'{metric}_recent_vs_historical'] = (recent_mean - historical_mean) / (historical_mean + 1e-6)
            
            # Overall engagement trend
            activity_slope = trends.get('total_activity_slope', 0)
            engagement_slope = trends.get('engagement_ratio_slope', 0)
            
            # Classify trend direction
            if activity_slope > 0.1 and engagement_slope > 0.01:
                trend_direction = 'INCREASING'
                risk_level = 'LOW'
            elif activity_slope < -0.1 or engagement_slope < -0.01:
                trend_direction = 'DECREASING'
                risk_level = 'HIGH' if activity_slope < -0.2 else 'MEDIUM'
            else:
                trend_direction = 'STABLE'
                risk_level = 'LOW'
            
            trends.update({
                'user_id': user_id,
                'trend_direction': trend_direction,
                'risk_level': risk_level,
                'data_points': len(user_ts),
                'analysis_period_days': (user_ts['date'].max() - user_ts['date'].min()).days
            })
            
            trend_results.append(trends)
        
        trends_df = pd.DataFrame(trend_results)
        
        print(f"Analyzed trends for {len(trends_df)} users")
        print(f"Trend distribution: {trends_df['trend_direction'].value_counts().to_dict()}")
        
        return trends_df
    
    def detect_anomalies(self, ts_df: pd.DataFrame, method: str = 'isolation_forest') -> pd.DataFrame:
        """Detect anomalous behavior patterns"""
        
        print(f"Detecting anomalies using {method}...")
        
        anomaly_results = []
        
        for user_id in ts_df['user_id'].unique():
            user_ts = ts_df[ts_df['user_id'] == user_id].sort_values('date')
            
            if len(user_ts) < 7:  # Need minimum data
                continue
            
            # Features for anomaly detection
            features = ['edit_count', 'meeting_count', 'engagement_ratio', 'edit_velocity', 'meeting_velocity']
            user_features = user_ts[features].fillna(0)
            
            if method == 'statistical':
                # Statistical anomaly detection using Z-score
                z_scores = np.abs(stats.zscore(user_features, axis=0))
                anomaly_scores = np.max(z_scores, axis=1)
                threshold = 2.5
                
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                
                if len(user_features) < 10:
                    # Use simpler method for small datasets
                    anomaly_scores = np.abs(stats.zscore(user_features.mean(axis=1)))
                    threshold = 2.0
                else:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_pred = iso_forest.fit_predict(user_features)
                    anomaly_scores = -iso_forest.score_samples(user_features)
                    threshold = np.percentile(anomaly_scores, 90)
            
            else:  # moving_average
                # Anomaly detection using moving average deviation
                window_size = min(7, len(user_features) // 2)
                rolling_mean = user_features.rolling(window=window_size, center=True).mean()
                deviations = np.abs(user_features - rolling_mean).mean(axis=1)
                anomaly_scores = deviations.fillna(0).values
                threshold = np.percentile(anomaly_scores, 80)
            
            # Identify anomalous periods
            anomalous_dates = user_ts[anomaly_scores > threshold]['date'].tolist()
            
            # Calculate anomaly statistics
            anomaly_count = len(anomalous_dates)
            max_anomaly_score = np.max(anomaly_scores) if len(anomaly_scores) > 0 else 0
            avg_anomaly_score = np.mean(anomaly_scores) if len(anomaly_scores) > 0 else 0
            
            # Recent anomaly trend
            recent_period = 14  # last 2 weeks
            recent_scores = anomaly_scores[-recent_period:] if len(anomaly_scores) >= recent_period else anomaly_scores
            recent_anomaly_trend = np.mean(recent_scores) if len(recent_scores) > 0 else 0
            
            anomaly_results.append({
                'user_id': user_id,
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_count / len(user_ts),
                'max_anomaly_score': max_anomaly_score,
                'avg_anomaly_score': avg_anomaly_score,
                'recent_anomaly_trend': recent_anomaly_trend,
                'anomalous_dates': anomalous_dates,
                'method_used': method
            })
        
        anomalies_df = pd.DataFrame(anomaly_results)
        
        print(f"Detected anomalies for {len(anomalies_df)} users")
        print(f"Average anomaly rate: {anomalies_df['anomaly_rate'].mean():.2%}")
        
        self.anomaly_scores = {row['user_id']: row['avg_anomaly_score'] for _, row in anomalies_df.iterrows()}
        
        return anomalies_df
    
    def analyze_seasonal_patterns(self, ts_df: pd.DataFrame) -> dict:
        """Analyze seasonal and weekly patterns"""
        
        print("Analyzing seasonal patterns...")
        
        patterns = {}
        
        # Weekly patterns
        weekly_pattern = ts_df.groupby('day_of_week').agg({
            'edit_count': 'mean',
            'meeting_count': 'mean',
            'total_activity': 'mean',
            'engagement_ratio': 'mean'
        })
        
        patterns['weekly'] = weekly_pattern.to_dict()
        
        # Weekend vs weekday patterns
        weekend_stats = ts_df[ts_df['is_weekend']].agg({
            'edit_count': 'mean',
            'meeting_count': 'mean',
            'total_activity': 'mean'
        })
        
        weekday_stats = ts_df[~ts_df['is_weekend']].agg({
            'edit_count': 'mean',
            'meeting_count': 'mean',
            'total_activity': 'mean'
        })
        
        patterns['weekend_vs_weekday'] = {
            'weekend': weekend_stats.to_dict(),
            'weekday': weekday_stats.to_dict(),
            'weekend_ratio': (weekend_stats / weekday_stats).to_dict()
        }
        
        # User-specific patterns
        user_patterns = {}
        for user_id in ts_df['user_id'].unique():
            user_ts = ts_df[ts_df['user_id'] == user_id]
            
            if len(user_ts) < 14:  # Need minimum data
                continue
            
            # Calculate user's weekly pattern strength
            user_weekly = user_ts.groupby('day_of_week')['total_activity'].mean()
            pattern_strength = user_weekly.std() / (user_weekly.mean() + 1e-6)
            
            # Peak activity day
            peak_day = user_weekly.idxmax()
            low_day = user_weekly.idxmin()
            
            user_patterns[user_id] = {
                'pattern_strength': pattern_strength,
                'peak_day': peak_day,
                'low_day': low_day,
                'weekend_preference': user_ts[user_ts['is_weekend']]['total_activity'].mean() / 
                                    (user_ts[~user_ts['is_weekend']]['total_activity'].mean() + 1e-6)
            }
        
        patterns['user_specific'] = user_patterns
        
        self.seasonal_patterns = patterns
        
        print(f"Analyzed patterns for {len(user_patterns)} users")
        
        return patterns
    
    def cluster_behavioral_patterns(self, ts_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster users based on temporal behavioral patterns"""
        
        print(f"Clustering behavioral patterns into {n_clusters} groups...")
        
        # Aggregate user-level features
        user_features = ts_df.groupby('user_id').agg({
            'edit_count': ['mean', 'std', 'max'],
            'meeting_count': ['mean', 'std', 'max'],
            'total_activity': ['mean', 'std', 'max'],
            'engagement_ratio': ['mean', 'std'],
            'edit_velocity': ['mean', 'std'],
            'meeting_velocity': ['mean', 'std'],
            'edit_pattern_score': 'mean',
            'meeting_pattern_score': 'mean'
        }).fillna(0)
        
        # Flatten column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns]
        
        # Add weekend preference
        weekend_preference = ts_df.groupby('user_id').apply(
            lambda x: x[x['is_weekend']]['total_activity'].mean() / 
                     (x[~x['is_weekend']]['total_activity'].mean() + 1e-6)
        ).fillna(1.0)
        
        user_features['weekend_preference'] = weekend_preference
        
        # Scale features
        features_scaled = self.scaler.fit_transform(user_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create results dataframe
        cluster_results = pd.DataFrame({
            'user_id': user_features.index,
            'cluster': cluster_labels
        })
        
        # Analyze cluster characteristics
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = user_features[cluster_mask]
            
            stats = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'avg_activity': cluster_features['total_activity_mean'].mean(),
                'avg_engagement': cluster_features['engagement_ratio_mean'].mean(),
                'activity_consistency': 1 / (cluster_features['total_activity_std'].mean() + 1),
                'weekend_preference': cluster_features['weekend_preference'].mean()
            }
            
            # Assign cluster interpretation
            if stats['avg_activity'] > user_features['total_activity_mean'].quantile(0.8):
                interpretation = 'High Engagement'
            elif stats['avg_activity'] < user_features['total_activity_mean'].quantile(0.2):
                interpretation = 'Low Engagement'
            elif stats['activity_consistency'] > 2:
                interpretation = 'Consistent Moderate'
            elif stats['weekend_preference'] > 1.5:
                interpretation = 'Weekend Focused'
            else:
                interpretation = 'Variable Engagement'
            
            stats['interpretation'] = interpretation
            cluster_stats.append(stats)
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        
        print("Cluster Analysis Results:")
        print(cluster_stats_df[['cluster_id', 'size', 'interpretation', 'avg_activity']].to_string(index=False))
        
        return cluster_results, cluster_stats_df, user_features
    
    def forecast_engagement_trends(self, ts_df: pd.DataFrame, forecast_days: int = 14) -> pd.DataFrame:
        """Simple trend forecasting for engagement metrics"""
        
        print(f"Forecasting engagement trends for next {forecast_days} days...")
        
        forecast_results = []
        
        for user_id in ts_df['user_id'].unique():
            user_ts = ts_df[ts_df['user_id'] == user_id].sort_values('date')
            
            if len(user_ts) < 10:  # Need minimum history
                continue
            
            # Simple linear trend extrapolation
            metrics = ['total_activity', 'engagement_ratio']
            
            for metric in metrics:
                values = user_ts[metric].values
                x = np.arange(len(values))
                
                # Fit linear trend
                slope, intercept, r_value, _, _ = stats.linregress(x, values)
                
                # Forecast future values
                future_x = np.arange(len(values), len(values) + forecast_days)
                forecast_values = slope * future_x + intercept
                
                # Calculate trend confidence based on R-squared
                confidence = r_value ** 2
                
                # Determine trend direction and risk
                avg_forecast = np.mean(forecast_values)
                current_avg = np.mean(values[-7:])  # last week average
                
                change_percent = (avg_forecast - current_avg) / (current_avg + 1e-6)
                
                if change_percent < -0.2:
                    trend_risk = 'HIGH'
                elif change_percent < -0.1:
                    trend_risk = 'MEDIUM'
                else:
                    trend_risk = 'LOW'
                
                forecast_results.append({
                    'user_id': user_id,
                    'metric': metric,
                    'current_value': values[-1],
                    'forecast_avg': avg_forecast,
                    'change_percent': change_percent,
                    'trend_confidence': confidence,
                    'trend_risk': trend_risk,
                    'forecast_values': forecast_values.tolist()
                })
        
        forecast_df = pd.DataFrame(forecast_results)
        
        print(f"Generated forecasts for {len(forecast_df['user_id'].unique())} users")
        
        return forecast_df
    
    def create_temporal_dashboard_data(self, ts_df: pd.DataFrame, trends_df: pd.DataFrame, 
                                     anomalies_df: pd.DataFrame) -> dict:
        """Prepare data for temporal analysis dashboard"""
        
        dashboard_data = {}
        
        # Overall trends
        dashboard_data['overall_trends'] = {
            'total_users': len(ts_df['user_id'].unique()),
            'avg_daily_activity': ts_df.groupby('date')['total_activity'].sum().mean(),
            'trend_distribution': trends_df['trend_direction'].value_counts().to_dict() if not trends_df.empty else {},
            'high_risk_users': len(trends_df[trends_df['risk_level'] == 'HIGH']) if not trends_df.empty else 0
        }
        
        # Weekly patterns
        weekly_agg = ts_df.groupby('day_of_week').agg({
            'total_activity': 'mean',
            'edit_count': 'mean',
            'meeting_count': 'mean'
        })
        
        dashboard_data['weekly_patterns'] = {
            'day_labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'activity_by_day': weekly_agg['total_activity'].tolist(),
            'edits_by_day': weekly_agg['edit_count'].tolist(),
            'meetings_by_day': weekly_agg['meeting_count'].tolist()
        }
        
        # Anomaly summary
        if not anomalies_df.empty:
            dashboard_data['anomaly_summary'] = {
                'users_with_anomalies': len(anomalies_df[anomalies_df['anomaly_count'] > 0]),
                'avg_anomaly_rate': anomalies_df['anomaly_rate'].mean(),
                'high_anomaly_users': len(anomalies_df[anomalies_df['anomaly_rate'] > 0.2])
            }
        else:
            dashboard_data['anomaly_summary'] = {
                'users_with_anomalies': 0,
                'avg_anomaly_rate': 0,
                'high_anomaly_users': 0
            }
        
        # Time series for charts
        daily_agg = ts_df.groupby('date').agg({
            'total_activity': 'sum',
            'user_id': 'nunique'  # active users per day
        }).reset_index()
        
        dashboard_data['time_series'] = {
            'dates': daily_agg['date'].dt.strftime('%Y-%m-%d').tolist(),
            'total_activity': daily_agg['total_activity'].tolist(),
            'active_users': daily_agg['user_id'].tolist()
        }
        
        return dashboard_data
    
    def plot_temporal_analysis(self, ts_df: pd.DataFrame, trends_df: pd.DataFrame = None, 
                             anomalies_df: pd.DataFrame = None):
        """Create comprehensive temporal analysis visualizations"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # 1. Overall activity trend
        daily_activity = ts_df.groupby('date')['total_activity'].sum()
        axes[0, 0].plot(daily_activity.index, daily_activity.values, linewidth=2, color='blue')
        axes[0, 0].set_title('Overall Daily Activity Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Activity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Weekly pattern heatmap
        weekly_data = ts_df.pivot_table(
            values='total_activity', 
            index='user_id', 
            columns='day_of_week', 
            aggfunc='mean'
        ).fillna(0)
        
        # Sample users for visualization
        sample_users = weekly_data.head(20)
        sns.heatmap(sample_users, cmap='YlOrRd', ax=axes[0, 1], cbar=True)
        axes[0, 1].set_title('Weekly Activity Patterns (Sample Users)')
        axes[0, 1].set_xlabel('Day of Week (0=Mon, 6=Sun)')
        axes[0, 1].set_ylabel('Users')
        
        # 3. Distribution of engagement metrics
        ts_df['engagement_ratio'].hist(bins=30, ax=axes[1, 0], alpha=0.7, color='green')
        axes[1, 0].set_title('Distribution of Engagement Ratios')
        axes[1, 0].set_xlabel('Engagement Ratio')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Trend distribution (if available)
        if trends_df is not None and not trends_df.empty:
            trend_counts = trends_df['trend_direction'].value_counts()
            axes[1, 1].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Distribution of Engagement Trends')
        else:
            axes[1, 1].text(0.5, 0.5, 'Trend data not available', 
                          horizontalalignment='center', verticalalignment='center')
            axes[1, 1].set_title('Trend Distribution')
        
        # 5. Activity by day of week
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_avg = ts_df.groupby('day_of_week')['total_activity'].mean()
        axes[2, 0].bar(range(7), weekly_avg.values, color='skyblue', alpha=0.7)
        axes[2, 0].set_xticks(range(7))
        axes[2, 0].set_xticklabels(day_names)
        axes[2, 0].set_title('Average Activity by Day of Week')
        axes[2, 0].set_ylabel('Average Activity')
        
        # 6. Anomaly scores distribution (if available)
        if anomalies_df is not None and not anomalies_df.empty:
            anomalies_df['avg_anomaly_score'].hist(bins=20, ax=axes[2, 1], alpha=0.7, color='red')
            axes[2, 1].set_title('Distribution of Anomaly Scores')
            axes[2, 1].set_xlabel('Anomaly Score')
            axes[2, 1].set_ylabel('Frequency')
        else:
            axes[2, 1].text(0.5, 0.5, 'Anomaly data not available', 
                          horizontalalignment='center', verticalalignment='center')
            axes[2, 1].set_title('Anomaly Score Distribution')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("synthetic_user_behavior_log.csv")
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer()
    
    # Create time series
    ts_df = analyzer.create_time_series(data, time_window_days=7)
    
    # Analyze trends
    trends_df = analyzer.detect_trends(ts_df)
    
    # Detect anomalies
    anomalies_df = analyzer.detect_anomalies(ts_df, method='statistical')
    
    # Analyze seasonal patterns
    seasonal_patterns = analyzer.analyze_seasonal_patterns(ts_df)
    
    # Cluster behavioral patterns
    cluster_results, cluster_stats, user_features = analyzer.cluster_behavioral_patterns(ts_df)
    
    # Forecast trends
    forecast_df = analyzer.forecast_engagement_trends(ts_df, forecast_days=14)
    
    # Display results
    print("\nTemporal Analysis Results:")
    print("=" * 50)
    print(f"Time series data points: {len(ts_df)}")
    print(f"Users analyzed: {len(ts_df['user_id'].unique())}")
    print(f"Date range: {ts_df['date'].min()} to {ts_df['date'].max()}")
    
    print(f"\nTrend Analysis:")
    if not trends_df.empty:
        print(trends_df['trend_direction'].value_counts())
        print(f"High-risk users: {len(trends_df[trends_df['risk_level'] == 'HIGH'])}")
    
    print(f"\nAnomaly Detection:")
    if not anomalies_df.empty:
        print(f"Users with anomalies: {len(anomalies_df[anomalies_df['anomaly_count'] > 0])}")
        print(f"Average anomaly rate: {anomalies_df['anomaly_rate'].mean():.1%}")
    
    print(f"\nBehavioral Clusters:")
    print(cluster_stats[['cluster_id', 'size', 'interpretation']].to_string(index=False))
    
    # Create visualizations
    analyzer.plot_temporal_analysis(ts_df, trends_df, anomalies_df)
    
    # Save results
    ts_df.to_csv('temporal_analysis_data.csv', index=False)
    if not trends_df.empty:
        trends_df.to_csv('engagement_trends.csv', index=False)
    if not anomalies_df.empty:
        anomalies_df.to_csv('behavioral_anomalies.csv', index=False)
    cluster_results.to_csv('behavioral_clusters.csv', index=False)
    
    print(f"\nResults saved to CSV files")
    
    # Prepare dashboard data
    dashboard_data = analyzer.create_temporal_dashboard_data(ts_df, trends_df, anomalies_df)
    
    import json
    with open('temporal_dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print(f"Dashboard data saved to temporal_dashboard_data.json")
