"""
Advanced Feature Engineering for Behavioral Signal Analysis
Sophisticated feature extraction and pattern recognition
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for behavioral analysis"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_history = {}
        
    def extract_temporal_features(self, timestamps_str: str, feature_prefix: str) -> Dict:
        """Extract comprehensive temporal features from timestamp data"""
        
        if pd.isna(timestamps_str) or str(timestamps_str).strip() == '':
            return self._get_empty_temporal_features(feature_prefix)
        
        try:
            times = [datetime.fromisoformat(t.strip()) for t in str(timestamps_str).split(",")]
            times.sort()
            
            if len(times) < 2:
                return self._get_minimal_temporal_features(times, feature_prefix)
            
            # Basic gap analysis
            gaps = [(times[i] - times[i-1]).total_seconds() / 3600 for i in range(1, len(times))]  # hours
            
            # Time-based features
            hours = [t.hour for t in times]
            days_of_week = [t.weekday() for t in times]
            
            # Advanced statistical features
            features = {
                f'{feature_prefix}_count': len(times),
                f'{feature_prefix}_span_days': (times[-1] - times[0]).days,
                f'{feature_prefix}_avg_gap_hours': np.mean(gaps),
                f'{feature_prefix}_std_gap_hours': np.std(gaps),
                f'{feature_prefix}_min_gap_hours': np.min(gaps),
                f'{feature_prefix}_max_gap_hours': np.max(gaps),
                f'{feature_prefix}_gap_coefficient_variation': np.std(gaps) / (np.mean(gaps) + 1e-6),
                
                # Temporal pattern features
                f'{feature_prefix}_avg_hour': np.mean(hours),
                f'{feature_prefix}_hour_std': np.std(hours),
                f'{feature_prefix}_weekend_ratio': sum(1 for d in days_of_week if d >= 5) / len(days_of_week),
                f'{feature_prefix}_business_hours_ratio': sum(1 for h in hours if 9 <= h <= 17) / len(hours),
                
                # Trend features
                f'{feature_prefix}_recent_activity_ratio': self._calculate_recent_activity_ratio(times),
                f'{feature_prefix}_acceleration': self._calculate_acceleration(gaps),
                f'{feature_prefix}_periodicity_score': self._calculate_periodicity(gaps),
                
                # Anomaly features
                f'{feature_prefix}_outlier_ratio': self._calculate_outlier_ratio(gaps),
                f'{feature_prefix}_burst_score': self._calculate_burst_score(times),
                f'{feature_prefix}_entropy': self._calculate_temporal_entropy(gaps),
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {feature_prefix} timestamps: {e}")
            return self._get_empty_temporal_features(feature_prefix)
    
    def _get_empty_temporal_features(self, prefix: str) -> Dict:
        """Return empty feature set"""
        return {f'{prefix}_{feature}': 0 for feature in [
            'count', 'span_days', 'avg_gap_hours', 'std_gap_hours', 'min_gap_hours', 
            'max_gap_hours', 'gap_coefficient_variation', 'avg_hour', 'hour_std',
            'weekend_ratio', 'business_hours_ratio', 'recent_activity_ratio',
            'acceleration', 'periodicity_score', 'outlier_ratio', 'burst_score', 'entropy'
        ]}
    
    def _get_minimal_temporal_features(self, times: List[datetime], prefix: str) -> Dict:
        """Return minimal features for single timestamp"""
        features = self._get_empty_temporal_features(prefix)
        if times:
            features[f'{prefix}_count'] = len(times)
            features[f'{prefix}_avg_hour'] = times[0].hour
            features[f'{prefix}_weekend_ratio'] = 1 if times[0].weekday() >= 5 else 0
            features[f'{prefix}_business_hours_ratio'] = 1 if 9 <= times[0].hour <= 17 else 0
        return features
    
    def _calculate_recent_activity_ratio(self, times: List[datetime], days_back: int = 14) -> float:
        """Calculate ratio of recent activity vs historical"""
        cutoff = datetime.now() - timedelta(days=days_back)
        recent_count = sum(1 for t in times if t >= cutoff)
        return recent_count / len(times) if times else 0
    
    def _calculate_acceleration(self, gaps: List[float]) -> float:
        """Calculate acceleration in gap changes"""
        if len(gaps) < 3:
            return 0
        
        gap_changes = [gaps[i] - gaps[i-1] for i in range(1, len(gaps))]
        return np.mean(gap_changes)
    
    def _calculate_periodicity(self, gaps: List[float]) -> float:
        """Calculate periodicity score using autocorrelation"""
        if len(gaps) < 5:
            return 0
        
        try:
            # Simple autocorrelation at lag 1
            correlation = np.corrcoef(gaps[:-1], gaps[1:])[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0
        except:
            return 0
    
    def _calculate_outlier_ratio(self, gaps: List[float]) -> float:
        """Calculate ratio of outlier gaps using IQR method"""
        if len(gaps) < 4:
            return 0
        
        q1, q3 = np.percentile(gaps, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = sum(1 for gap in gaps if gap < lower_bound or gap > upper_bound)
        return outliers / len(gaps)
    
    def _calculate_burst_score(self, times: List[datetime]) -> float:
        """Calculate burstiness score (tendency for events to cluster)"""
        if len(times) < 3:
            return 0
        
        gaps = [(times[i] - times[i-1]).total_seconds() for i in range(1, len(times))]
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        if mean_gap == 0:
            return 0
        
        # Burstiness parameter from literature
        burstiness = (std_gap - mean_gap) / (std_gap + mean_gap)
        return max(-1, min(1, burstiness))  # Normalize to [-1, 1]
    
    def _calculate_temporal_entropy(self, gaps: List[float]) -> float:
        """Calculate entropy of gap distribution"""
        if len(gaps) < 2:
            return 0
        
        try:
            # Bin the gaps and calculate entropy
            hist, _ = np.histogram(gaps, bins=min(10, len(gaps)))
            hist = hist[hist > 0]  # Remove zero bins
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        except:
            return 0
    
    def extract_calendar_void_features(self, voids_str: str) -> Dict:
        """Extract features from calendar void data"""
        
        if pd.isna(voids_str) or str(voids_str).strip() == '':
            return {
                'void_total_days': 0,
                'void_max_duration': 0,
                'void_frequency': 0,
                'void_coverage_ratio': 0,
                'void_pattern_score': 0
            }
        
        try:
            void_str = str(voids_str).strip()
            parts = void_str.split(':')
            
            if len(parts) >= 6:
                start_date_str = ':'.join(parts[:3])
                end_date_str = ':'.join(parts[3:6])
                
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
                void_days = max((end_date - start_date).days, 0)
            else:
                void_days = 0
            
            # Calculate additional void metrics
            analysis_period = 60  # days
            coverage_ratio = min(void_days / analysis_period, 1.0)
            
            # Pattern scoring (placeholder for more complex analysis)
            pattern_score = self._calculate_void_pattern_score(void_days, analysis_period)
            
            return {
                'void_total_days': void_days,
                'void_max_duration': void_days,  # Simplified - could track multiple voids
                'void_frequency': 1 if void_days > 0 else 0,
                'void_coverage_ratio': coverage_ratio,
                'void_pattern_score': pattern_score
            }
            
        except Exception as e:
            print(f"Error processing calendar voids: {e}")
            return {
                'void_total_days': 0,
                'void_max_duration': 0,
                'void_frequency': 0,
                'void_coverage_ratio': 0,
                'void_pattern_score': 0
            }
    
    def _calculate_void_pattern_score(self, void_days: int, period: int) -> float:
        """Calculate void pattern anomaly score"""
        if void_days == 0:
            return 0
        
        # Score based on deviation from expected patterns
        expected_void_ratio = 0.1  # 10% expected void time
        actual_ratio = void_days / period
        
        # Higher score for more unusual patterns
        deviation = abs(actual_ratio - expected_void_ratio)
        return min(deviation * 2, 1.0)
    
    def extract_interaction_features(self, edit_features: Dict, meeting_features: Dict, 
                                   void_features: Dict) -> Dict:
        """Extract interaction features between different behavioral signals"""
        
        interactions = {}
        
        # Edit-Meeting interactions
        edit_count = edit_features.get('edit_count', 0)
        meeting_count = meeting_features.get('meeting_count', 0)
        
        interactions['edit_to_meeting_ratio'] = edit_count / (meeting_count + 1)
        interactions['meeting_to_edit_ratio'] = meeting_count / (edit_count + 1)
        
        # Activity synchronization
        edit_gap = edit_features.get('edit_avg_gap_hours', 0)
        meeting_span = meeting_features.get('meeting_span_days', 0)
        
        interactions['activity_synchronization'] = 1 / (abs(edit_gap/24 - meeting_span) + 1)
        
        # Void impact on activity
        void_ratio = void_features.get('void_coverage_ratio', 0)
        total_activity = edit_count + meeting_count
        
        interactions['void_activity_impact'] = void_ratio * total_activity
        interactions['activity_void_resistance'] = total_activity / (void_ratio + 1)
        
        # Temporal alignment
        edit_hour_std = edit_features.get('edit_hour_std', 0)
        meeting_hour_std = meeting_features.get('meeting_hour_std', 0)
        
        interactions['temporal_consistency'] = 1 / (edit_hour_std + meeting_hour_std + 1)
        
        # Behavioral diversity
        edit_entropy = edit_features.get('edit_entropy', 0)
        meeting_entropy = meeting_features.get('meeting_entropy', 0)
        
        interactions['behavioral_diversity'] = (edit_entropy + meeting_entropy) / 2
        
        return interactions
    
    def calculate_composite_scores(self, all_features: Dict) -> Dict:
        """Calculate composite behavioral scores"""
        
        composite = {}
        
        # Engagement consistency score
        edit_consistency = 1 / (all_features.get('edit_gap_coefficient_variation', 1) + 1)
        meeting_consistency = 1 / (all_features.get('meeting_gap_coefficient_variation', 1) + 1)
        composite['engagement_consistency'] = (edit_consistency + meeting_consistency) / 2
        
        # Activity intensity score
        edit_frequency = all_features.get('edit_count', 0) / 30  # per day approximation
        meeting_frequency = all_features.get('meeting_count', 0) / 30
        composite['activity_intensity'] = (edit_frequency + meeting_frequency) / 2
        
        # Temporal adaptability score
        weekend_activity = (all_features.get('edit_weekend_ratio', 0) + 
                          all_features.get('meeting_weekend_ratio', 0)) / 2
        business_hours_activity = (all_features.get('edit_business_hours_ratio', 0) + 
                                 all_features.get('meeting_business_hours_ratio', 0)) / 2
        composite['temporal_adaptability'] = abs(weekend_activity - 0.2) + abs(business_hours_activity - 0.8)
        
        # Disengagement risk composite
        void_impact = all_features.get('void_coverage_ratio', 0)
        activity_decline = 1 - all_features.get('edit_recent_activity_ratio', 1)
        pattern_disruption = all_features.get('edit_outlier_ratio', 0)
        
        composite['disengagement_risk_composite'] = (void_impact * 0.4 + 
                                                   activity_decline * 0.4 + 
                                                   pattern_disruption * 0.2)
        
        return composite
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        
        print("Starting advanced feature engineering...")
        
        engineered_data = data.copy()
        all_features = []
        
        for idx, row in data.iterrows():
            # Extract temporal features for each signal type
            edit_features = self.extract_temporal_features(
                row.get('message_edit_timestamps', ''), 'edit'
            )
            
            meeting_features = self.extract_temporal_features(
                row.get('meeting_timestamps', ''), 'meeting'
            )
            
            # Extract calendar void features
            void_features = self.extract_calendar_void_features(
                row.get('calendar_voids', '')
            )
            
            # Extract interaction features
            interaction_features = self.extract_interaction_features(
                edit_features, meeting_features, void_features
            )
            
            # Combine all features
            user_features = {**edit_features, **meeting_features, **void_features, **interaction_features}
            
            # Calculate composite scores
            composite_features = self.calculate_composite_scores(user_features)
            user_features.update(composite_features)
            
            # Add user ID
            user_features['user_id'] = row.get('user_id', f'User_{idx}')
            
            all_features.append(user_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Set user_id as index and move to first column
        if 'user_id' in features_df.columns:
            user_ids = features_df['user_id']
            features_df = features_df.drop('user_id', axis=1)
            features_df.insert(0, 'user_id', user_ids)
        
        print(f"Feature engineering complete. Generated {len(features_df.columns)-1} features for {len(features_df)} users.")
        
        return features_df
    
    def get_feature_importance_ranking(self, features_df: pd.DataFrame, 
                                     target_column: str = 'disengagement_risk_composite') -> pd.DataFrame:
        """Calculate and rank feature importance"""
        
        if target_column not in features_df.columns:
            print(f"Target column '{target_column}' not found. Using correlation analysis.")
            target_column = 'edit_count'  # Fallback
        
        # Calculate correlations with target
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        if target_column in numeric_features.columns:
            correlations = numeric_features.corr()[target_column].abs().sort_values(ascending=False)
            
            importance_df = pd.DataFrame({
                'feature': correlations.index,
                'importance_score': correlations.values,
                'feature_type': [self._categorize_feature(feat) for feat in correlations.index]
            })
            
            return importance_df[importance_df['feature'] != target_column]
        
        return pd.DataFrame()
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize features by type"""
        if 'edit' in feature_name:
            return 'Message Editing'
        elif 'meeting' in feature_name:
            return 'Meeting Patterns'
        elif 'void' in feature_name:
            return 'Calendar Voids'
        elif any(word in feature_name for word in ['ratio', 'interaction', 'synchronization']):
            return 'Interaction'
        elif any(word in feature_name for word in ['composite', 'consistency', 'intensity']):
            return 'Composite'
        else:
            return 'Other'

# Example usage and testing
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("synthetic_user_behavior_log.csv")
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Engineer features
    features_df = engineer.engineer_features(data)
    
    # Display results
    print("\nGenerated Features:")
    print("=" * 50)
    print(f"Shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns[1:])}")  # Exclude user_id
    
    # Show sample of engineered features
    print("\nSample Features (first 5 users):")
    display_columns = ['user_id'] + [col for col in features_df.columns if 'composite' in col or 'ratio' in col][:5]
    print(features_df[display_columns].head())
    
    # Feature importance analysis
    importance_df = engineer.get_feature_importance_ranking(features_df)
    if not importance_df.empty:
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Feature type distribution
        type_counts = importance_df['feature_type'].value_counts()
        print(f"\nFeature Type Distribution:")
        for ftype, count in type_counts.items():
            print(f"{ftype}: {count} features")
    
    # Save engineered features
    output_file = "engineered_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"\nEngineered features saved to {output_file}")
