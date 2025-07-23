"""
Executive Summary and System Integration Module
Comprehensive reporting and integration of all behavioral analysis components
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

class ExecutiveSummaryGenerator:
    """Generate comprehensive executive summaries and integrate all analysis components"""
    
    def __init__(self):
        self.analysis_results = {}
        self.summary_data = {}
        self.recommendations = []
        self.risk_matrix = {}
        
    def load_analysis_results(self, data_dir: str = ".") -> bool:
        """Load results from all analysis modules"""
        
        print("Loading analysis results...")
        
        try:
            # Basic data
            if os.path.exists(os.path.join(data_dir, "synthetic_user_behavior_log.csv")):
                self.analysis_results['raw_data'] = pd.read_csv(
                    os.path.join(data_dir, "synthetic_user_behavior_log.csv")
                )
            
            # Engineered features
            if os.path.exists(os.path.join(data_dir, "engineered_features.csv")):
                self.analysis_results['features'] = pd.read_csv(
                    os.path.join(data_dir, "engineered_features.csv")
                )
            
            # Model optimization results
            if os.path.exists(os.path.join(data_dir, "model_optimization_results.csv")):
                self.analysis_results['model_results'] = pd.read_csv(
                    os.path.join(data_dir, "model_optimization_results.csv")
                )
            
            # Temporal analysis
            if os.path.exists(os.path.join(data_dir, "temporal_analysis_data.csv")):
                self.analysis_results['temporal_data'] = pd.read_csv(
                    os.path.join(data_dir, "temporal_analysis_data.csv")
                )
            
            if os.path.exists(os.path.join(data_dir, "engagement_trends.csv")):
                self.analysis_results['trends'] = pd.read_csv(
                    os.path.join(data_dir, "engagement_trends.csv")
                )
            
            if os.path.exists(os.path.join(data_dir, "behavioral_anomalies.csv")):
                self.analysis_results['anomalies'] = pd.read_csv(
                    os.path.join(data_dir, "behavioral_anomalies.csv")
                )
            
            if os.path.exists(os.path.join(data_dir, "behavioral_clusters.csv")):
                self.analysis_results['clusters'] = pd.read_csv(
                    os.path.join(data_dir, "behavioral_clusters.csv")
                )
            
            # Alerts data
            if os.path.exists(os.path.join(data_dir, "disengagement_alerts.json")):
                with open(os.path.join(data_dir, "disengagement_alerts.json"), 'r') as f:
                    self.analysis_results['alerts'] = json.load(f)
            
            print(f"Loaded {len(self.analysis_results)} analysis components")
            return True
            
        except Exception as e:
            print(f"Error loading analysis results: {e}")
            return False
    
    def calculate_system_metrics(self) -> Dict:
        """Calculate high-level system metrics"""
        
        metrics = {
            'system_health': {},
            'user_engagement': {},
            'risk_assessment': {},
            'model_performance': {},
            'temporal_insights': {}
        }
        
        # System health metrics
        if 'raw_data' in self.analysis_results:
            raw_data = self.analysis_results['raw_data']
            metrics['system_health'] = {
                'total_users': len(raw_data),
                'data_quality_score': self._calculate_data_quality(raw_data),
                'coverage_percentage': 100.0,  # Assuming full coverage
                'last_updated': datetime.now().isoformat()
            }
        
        # User engagement metrics
        if 'temporal_data' in self.analysis_results:
            temporal_data = self.analysis_results['temporal_data']
            metrics['user_engagement'] = {
                'avg_daily_activity': temporal_data['total_activity'].mean(),
                'engagement_trend': self._calculate_engagement_trend(temporal_data),
                'active_user_percentage': self._calculate_active_users(temporal_data),
                'engagement_consistency': self._calculate_consistency(temporal_data)
            }
        
        # Risk assessment metrics
        risk_metrics = self._calculate_risk_metrics()
        metrics['risk_assessment'] = risk_metrics
        
        # Model performance metrics
        if 'model_results' in self.analysis_results:
            model_results = self.analysis_results['model_results']
            best_model = model_results.loc[model_results['F1_Score'].idxmax()]
            metrics['model_performance'] = {
                'best_model': best_model['Model'],
                'best_f1_score': best_model['F1_Score'],
                'best_accuracy': best_model['Accuracy'],
                'model_confidence': self._calculate_model_confidence(model_results)
            }
        
        # Temporal insights
        if 'trends' in self.analysis_results:
            trends = self.analysis_results['trends']
            metrics['temporal_insights'] = {
                'users_declining': len(trends[trends['trend_direction'] == 'DECREASING']),
                'users_improving': len(trends[trends['trend_direction'] == 'INCREASING']),
                'stability_percentage': len(trends[trends['trend_direction'] == 'STABLE']) / len(trends) * 100
            }
        
        self.summary_data['metrics'] = metrics
        return metrics
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        quality_score = (total_cells - missing_cells) / total_cells * 100
        return round(quality_score, 2)
    
    def _calculate_engagement_trend(self, temporal_data: pd.DataFrame) -> str:
        """Calculate overall engagement trend"""
        if 'date' not in temporal_data.columns:
            return 'UNKNOWN'
        
        # Convert date column if it's string
        if temporal_data['date'].dtype == 'object':
            temporal_data = temporal_data.copy()
            temporal_data['date'] = pd.to_datetime(temporal_data['date'])
        
        daily_activity = temporal_data.groupby('date')['total_activity'].sum()
        
        if len(daily_activity) < 7:
            return 'INSUFFICIENT_DATA'
        
        # Calculate trend over last week vs previous week
        recent_avg = daily_activity.tail(7).mean()
        previous_avg = daily_activity.tail(14).head(7).mean()
        
        change_percent = (recent_avg - previous_avg) / (previous_avg + 1e-6)
        
        if change_percent > 0.1:
            return 'IMPROVING'
        elif change_percent < -0.1:
            return 'DECLINING'
        else:
            return 'STABLE'
    
    def _calculate_active_users(self, temporal_data: pd.DataFrame) -> float:
        """Calculate percentage of active users"""
        if 'user_id' not in temporal_data.columns:
            return 0.0
        
        recent_period = datetime.now() - timedelta(days=7)
        
        if 'date' in temporal_data.columns:
            if temporal_data['date'].dtype == 'object':
                temporal_data = temporal_data.copy()
                temporal_data['date'] = pd.to_datetime(temporal_data['date'])
            
            recent_data = temporal_data[temporal_data['date'] >= recent_period]
            active_users = len(recent_data[recent_data['total_activity'] > 0]['user_id'].unique())
            total_users = len(temporal_data['user_id'].unique())
        else:
            active_users = len(temporal_data[temporal_data['total_activity'] > 0]['user_id'].unique())
            total_users = len(temporal_data['user_id'].unique())
        
        return round(active_users / total_users * 100, 2) if total_users > 0 else 0.0
    
    def _calculate_consistency(self, temporal_data: pd.DataFrame) -> float:
        """Calculate engagement consistency score"""
        if 'user_id' not in temporal_data.columns or 'total_activity' not in temporal_data.columns:
            return 0.0
        
        user_std = temporal_data.groupby('user_id')['total_activity'].std().fillna(0)
        user_mean = temporal_data.groupby('user_id')['total_activity'].mean()
        
        # Coefficient of variation (lower is more consistent)
        cv = user_std / (user_mean + 1e-6)
        consistency_score = 1 / (cv.mean() + 1)  # Inverse for higher = better
        
        return round(min(consistency_score * 10, 100), 2)  # Scale to 0-100
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics"""
        risk_metrics = {
            'high_risk_users': 0,
            'medium_risk_users': 0,
            'low_risk_users': 0,
            'critical_alerts': 0,
            'overall_risk_level': 'LOW'
        }
        
        # From alerts data
        if 'alerts' in self.analysis_results:
            alerts = self.analysis_results['alerts']
            
            risk_levels = [alert.get('risk_level', 'LOW') for alert in alerts]
            risk_metrics['critical_alerts'] = risk_levels.count('CRITICAL')
            risk_metrics['high_risk_users'] = risk_levels.count('HIGH')
            risk_metrics['medium_risk_users'] = risk_levels.count('MEDIUM')
            risk_metrics['low_risk_users'] = risk_levels.count('LOW')
            
            # Overall risk level
            if risk_metrics['critical_alerts'] > 0:
                risk_metrics['overall_risk_level'] = 'CRITICAL'
            elif risk_metrics['high_risk_users'] > len(alerts) * 0.2:
                risk_metrics['overall_risk_level'] = 'HIGH'
            elif risk_metrics['medium_risk_users'] > len(alerts) * 0.3:
                risk_metrics['overall_risk_level'] = 'MEDIUM'
            else:
                risk_metrics['overall_risk_level'] = 'LOW'
        
        # From trends data
        elif 'trends' in self.analysis_results:
            trends = self.analysis_results['trends']
            
            if 'risk_level' in trends.columns:
                risk_counts = trends['risk_level'].value_counts()
                risk_metrics['high_risk_users'] = risk_counts.get('HIGH', 0)
                risk_metrics['medium_risk_users'] = risk_counts.get('MEDIUM', 0)
                risk_metrics['low_risk_users'] = risk_counts.get('LOW', 0)
                
                total_users = len(trends)
                if risk_metrics['high_risk_users'] > total_users * 0.2:
                    risk_metrics['overall_risk_level'] = 'HIGH'
                elif risk_metrics['medium_risk_users'] > total_users * 0.3:
                    risk_metrics['overall_risk_level'] = 'MEDIUM'
        
        return risk_metrics
    
    def _calculate_model_confidence(self, model_results: pd.DataFrame) -> float:
        """Calculate overall model confidence"""
        if 'F1_Score' not in model_results.columns:
            return 0.0
        
        best_f1 = model_results['F1_Score'].max()
        model_variance = model_results['F1_Score'].std()
        
        # Higher F1 and lower variance = higher confidence
        confidence = best_f1 * (1 - min(model_variance, 0.5))
        return round(confidence * 100, 2)
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        metrics = self.summary_data.get('metrics', {})
        
        # Risk-based recommendations
        risk_metrics = metrics.get('risk_assessment', {})
        
        if risk_metrics.get('overall_risk_level') == 'CRITICAL':
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Risk Management',
                'title': 'Immediate Intervention Required',
                'description': 'Critical disengagement risk detected. Implement immediate intervention protocols.',
                'actions': [
                    'Schedule emergency team meetings',
                    'Conduct individual assessments',
                    'Review workload distribution',
                    'Assess team dynamics and conflicts'
                ]
            })
        
        elif risk_metrics.get('high_risk_users', 0) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Risk Management',
                'title': 'Address High-Risk Users',
                'description': f"{risk_metrics['high_risk_users']} users identified with high disengagement risk.",
                'actions': [
                    'Increase check-in frequency for high-risk users',
                    'Review individual performance and satisfaction',
                    'Consider workload adjustments',
                    'Provide additional support and resources'
                ]
            })
        
        # Engagement trend recommendations
        engagement = metrics.get('user_engagement', {})
        
        if engagement.get('engagement_trend') == 'DECLINING':
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Engagement',
                'title': 'Address Declining Engagement',
                'description': 'Overall team engagement is declining.',
                'actions': [
                    'Analyze root causes of decline',
                    'Implement engagement improvement initiatives',
                    'Review team processes and workload',
                    'Consider team building activities'
                ]
            })
        
        if engagement.get('engagement_consistency', 0) < 50:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Process Improvement',
                'title': 'Improve Engagement Consistency',
                'description': 'User engagement patterns are inconsistent.',
                'actions': [
                    'Establish regular communication patterns',
                    'Implement structured check-in processes',
                    'Provide clearer expectations and guidelines',
                    'Consider flexible work arrangements'
                ]
            })
        
        # Model performance recommendations
        model_performance = metrics.get('model_performance', {})
        
        if model_performance.get('model_confidence', 0) < 70:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'System Improvement',
                'title': 'Enhance Model Performance',
                'description': 'Detection model confidence could be improved.',
                'actions': [
                    'Collect additional training data',
                    'Implement advanced feature engineering',
                    'Consider ensemble modeling approaches',
                    'Regular model retraining schedule'
                ]
            })
        
        # Temporal pattern recommendations
        temporal_insights = metrics.get('temporal_insights', {})
        
        if temporal_insights.get('stability_percentage', 100) < 60:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Organizational',
                'title': 'Address Engagement Volatility',
                'description': 'High volatility in team engagement patterns detected.',
                'actions': [
                    'Investigate causes of engagement fluctuations',
                    'Implement stabilizing processes',
                    'Review organizational changes and their impact',
                    'Establish baseline engagement expectations'
                ]
            })
        
        # System health recommendations
        system_health = metrics.get('system_health', {})
        
        if system_health.get('data_quality_score', 100) < 90:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Data Quality',
                'title': 'Improve Data Collection',
                'description': 'Data quality issues detected in behavioral signals.',
                'actions': [
                    'Review data collection processes',
                    'Implement data validation checks',
                    'Train users on proper data entry',
                    'Automate data collection where possible'
                ]
            })
        
        # Default positive recommendation
        if not recommendations:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Maintenance',
                'title': 'Continue Monitoring',
                'description': 'System is operating within normal parameters.',
                'actions': [
                    'Maintain current monitoring frequency',
                    'Continue regular assessment cycles',
                    'Review and update thresholds periodically',
                    'Plan for proactive improvements'
                ]
            })
        
        self.recommendations = recommendations
        return recommendations
    
    def create_executive_dashboard_data(self) -> Dict:
        """Create data structure for executive dashboard"""
        
        dashboard_data = {
            'summary': {
                'generated_at': datetime.now().isoformat(),
                'analysis_period': self._get_analysis_period(),
                'system_status': self._get_system_status()
            },
            'key_metrics': self.summary_data.get('metrics', {}),
            'risk_matrix': self._create_risk_matrix(),
            'recommendations': self.recommendations,
            'alerts': self._get_active_alerts(),
            'trends': self._get_trend_summary()
        }
        
        return dashboard_data
    
    def _get_analysis_period(self) -> Dict:
        """Get analysis period information"""
        if 'temporal_data' in self.analysis_results:
            temporal_data = self.analysis_results['temporal_data']
            
            if 'date' in temporal_data.columns:
                if temporal_data['date'].dtype == 'object':
                    temporal_data = temporal_data.copy()
                    temporal_data['date'] = pd.to_datetime(temporal_data['date'])
                
                return {
                    'start_date': temporal_data['date'].min().isoformat(),
                    'end_date': temporal_data['date'].max().isoformat(),
                    'total_days': (temporal_data['date'].max() - temporal_data['date'].min()).days
                }
        
        return {
            'start_date': '2025-06-01',
            'end_date': '2025-07-23',
            'total_days': 52
        }
    
    def _get_system_status(self) -> str:
        """Determine overall system status"""
        metrics = self.summary_data.get('metrics', {})
        risk_level = metrics.get('risk_assessment', {}).get('overall_risk_level', 'LOW')
        
        if risk_level == 'CRITICAL':
            return 'CRITICAL'
        elif risk_level == 'HIGH':
            return 'WARNING'
        elif risk_level == 'MEDIUM':
            return 'CAUTION'
        else:
            return 'NORMAL'
    
    def _create_risk_matrix(self) -> Dict:
        """Create risk assessment matrix"""
        risk_matrix = {
            'users_by_risk': {},
            'risk_factors': {},
            'mitigation_status': {}
        }
        
        if 'alerts' in self.analysis_results:
            alerts = self.analysis_results['alerts']
            
            # Count users by risk level
            risk_levels = [alert.get('risk_level', 'LOW') for alert in alerts]
            for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                risk_matrix['users_by_risk'][level] = risk_levels.count(level)
            
            # Aggregate risk factors
            all_factors = []
            for alert in alerts:
                all_factors.extend(alert.get('contributing_factors', []))
            
            factor_counts = {}
            for factor in all_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            risk_matrix['risk_factors'] = dict(sorted(factor_counts.items(), 
                                                    key=lambda x: x[1], reverse=True)[:10])
        
        self.risk_matrix = risk_matrix
        return risk_matrix
    
    def _get_active_alerts(self) -> List[Dict]:
        """Get current active alerts"""
        if 'alerts' in self.analysis_results:
            # Filter for high priority alerts
            alerts = self.analysis_results['alerts']
            high_priority = [alert for alert in alerts 
                           if alert.get('risk_level') in ['HIGH', 'CRITICAL']]
            return high_priority[:10]  # Top 10 alerts
        
        return []
    
    def _get_trend_summary(self) -> Dict:
        """Get trend summary data"""
        if 'trends' in self.analysis_results:
            trends = self.analysis_results['trends']
            
            return {
                'improving_users': len(trends[trends['trend_direction'] == 'INCREASING']),
                'declining_users': len(trends[trends['trend_direction'] == 'DECREASING']),
                'stable_users': len(trends[trends['trend_direction'] == 'STABLE']),
                'total_analyzed': len(trends)
            }
        
        return {}
    
    def generate_executive_report(self, output_file: str = 'executive_report.html') -> str:
        """Generate comprehensive executive report"""
        
        # Calculate metrics and recommendations
        metrics = self.calculate_system_metrics()
        recommendations = self.generate_recommendations()
        
        # Create HTML report
        html_content = self._create_html_report(metrics, recommendations)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Executive report generated: {output_file}")
        
        return html_content
    
    def _create_html_report(self, metrics: Dict, recommendations: List[Dict]) -> str:
        """Create HTML executive report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Behavioral Signal Mapping - Executive Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; text-align: center; }}
                .risk-critical {{ background-color: #e74c3c; color: white; }}
                .risk-high {{ background-color: #e67e22; color: white; }}
                .risk-medium {{ background-color: #f39c12; color: white; }}
                .risk-low {{ background-color: #27ae60; color: white; }}
                .recommendation {{ margin: 10px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #ecf0f1; }}
                .recommendation.critical {{ border-left-color: #e74c3c; }}
                .recommendation.high {{ border-left-color: #e67e22; }}
                .recommendation.medium {{ border-left-color: #f39c12; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Behavioral Signal Mapping</h1>
                <h2>Executive Summary Report</h2>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>🎯 System Overview</h2>
                <div class="metric">
                    <h3>{metrics.get('system_health', {}).get('total_users', 'N/A')}</h3>
                    <p>Total Users</p>
                </div>
                <div class="metric">
                    <h3>{metrics.get('system_health', {}).get('data_quality_score', 'N/A')}%</h3>
                    <p>Data Quality</p>
                </div>
                <div class="metric">
                    <h3>{metrics.get('user_engagement', {}).get('active_user_percentage', 'N/A')}%</h3>
                    <p>Active Users</p>
                </div>
                <div class="metric">
                    <h3>{metrics.get('model_performance', {}).get('model_confidence', 'N/A')}%</h3>
                    <p>Model Confidence</p>
                </div>
            </div>
            
            <div class="section">
                <h2>⚠️ Risk Assessment</h2>
                <div class="metric risk-critical">
                    <h3>{metrics.get('risk_assessment', {}).get('critical_alerts', 0)}</h3>
                    <p>Critical Alerts</p>
                </div>
                <div class="metric risk-high">
                    <h3>{metrics.get('risk_assessment', {}).get('high_risk_users', 0)}</h3>
                    <p>High Risk Users</p>
                </div>
                <div class="metric risk-medium">
                    <h3>{metrics.get('risk_assessment', {}).get('medium_risk_users', 0)}</h3>
                    <p>Medium Risk Users</p>
                </div>
                <div class="metric risk-low">
                    <h3>{metrics.get('risk_assessment', {}).get('low_risk_users', 0)}</h3>
                    <p>Low Risk Users</p>
                </div>
                <p><strong>Overall Risk Level:</strong> <span class="risk-{metrics.get('risk_assessment', {}).get('overall_risk_level', 'low').lower()}">{metrics.get('risk_assessment', {}).get('overall_risk_level', 'LOW')}</span></p>
            </div>
            
            <div class="section">
                <h2>📈 Engagement Trends</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>Engagement Trend</td>
                        <td>{metrics.get('user_engagement', {}).get('engagement_trend', 'N/A')}</td>
                        <td>{'🔴' if metrics.get('user_engagement', {}).get('engagement_trend') == 'DECLINING' else '🟢' if metrics.get('user_engagement', {}).get('engagement_trend') == 'IMPROVING' else '🟡'}</td>
                    </tr>
                    <tr>
                        <td>Average Daily Activity</td>
                        <td>{str(metrics.get('user_engagement', {}).get('avg_daily_activity', 'N/A'))[:6]}</td>
                        <td>🟢</td>
                    </tr>
                    <tr>
                        <td>Engagement Consistency</td>
                        <td>{metrics.get('user_engagement', {}).get('engagement_consistency', 'N/A')}%</td>
                        <td>{'🟢' if metrics.get('user_engagement', {}).get('engagement_consistency', 0) > 70 else '🟡' if metrics.get('user_engagement', {}).get('engagement_consistency', 0) > 50 else '🔴'}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🔧 Recommendations</h2>
        """
        
        for rec in recommendations:
            priority_class = rec['priority'].lower()
            html += f"""
                <div class="recommendation {priority_class}">
                    <h3>{rec['title']} ({rec['priority']} Priority)</h3>
                    <p><strong>Category:</strong> {rec['category']}</p>
                    <p>{rec['description']}</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
            """
            for action in rec['actions']:
                html += f"<li>{action}</li>"
            html += "</ul></div>"
        
        html += f"""
            </div>
            
            <div class="section">
                <h2>📊 Model Performance</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Best Model</td>
                        <td>{metrics.get('model_performance', {}).get('best_model', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>{str(metrics.get('model_performance', {}).get('best_f1_score', 'N/A'))[:6]}</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{str(metrics.get('model_performance', {}).get('best_accuracy', 'N/A'))[:6]}</td>
                    </tr>
                    <tr>
                        <td>Model Confidence</td>
                        <td>{metrics.get('model_performance', {}).get('model_confidence', 'N/A')}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📋 Next Steps</h2>
                <ol>
                    <li><strong>Immediate Actions:</strong> Address any critical or high-priority recommendations</li>
                    <li><strong>Short-term (1-2 weeks):</strong> Implement process improvements and increase monitoring</li>
                    <li><strong>Medium-term (1-3 months):</strong> Evaluate system effectiveness and adjust parameters</li>
                    <li><strong>Long-term (3+ months):</strong> Strategic improvements and system enhancements</li>
                </ol>
            </div>
            
            <div class="section">
                <h2>ℹ️ System Information</h2>
                <p><strong>Analysis Period:</strong> {self._get_analysis_period().get('start_date', 'N/A')} to {self._get_analysis_period().get('end_date', 'N/A')}</p>
                <p><strong>Data Sources:</strong> Message edit timestamps, meeting patterns, calendar voids</p>
                <p><strong>Analysis Methods:</strong> Multi-agent behavioral analysis, machine learning classification, temporal pattern detection</p>
                <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html

# Example usage and integration
if __name__ == "__main__":
    # Initialize executive summary generator
    generator = ExecutiveSummaryGenerator()
    
    # Load all analysis results
    success = generator.load_analysis_results()
    
    if success:
        print("Analysis results loaded successfully")
        
        # Generate metrics and recommendations
        metrics = generator.calculate_system_metrics()
        recommendations = generator.generate_recommendations()
        
        # Print summary to console
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        
        print(f"\nSYSTEM HEALTH:")
        system_health = metrics.get('system_health', {})
        for key, value in system_health.items():
            print(f"  {key}: {value}")
        
        print(f"\nRISK ASSESSMENT:")
        risk_assessment = metrics.get('risk_assessment', {})
        for key, value in risk_assessment.items():
            print(f"  {key}: {value}")
        
        print(f"\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['title']} ({rec['priority']})")
        
        # Generate executive report
        generator.generate_executive_report()
        
        # Create dashboard data
        dashboard_data = generator.create_executive_dashboard_data()
        
        # Save dashboard data
        with open('executive_dashboard_data.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        print(f"\nExecutive dashboard data saved to executive_dashboard_data.json")
        print(f"Executive report saved to executive_report.html")
        
    else:
        print("Could not load analysis results. Please run the other analysis modules first.")
        
        # Create sample report with mock data
        print("Generating sample report with mock data...")
        
        # Mock metrics
        mock_metrics = {
            'system_health': {
                'total_users': 1000,
                'data_quality_score': 95.5,
                'coverage_percentage': 100.0,
                'last_updated': datetime.now().isoformat()
            },
            'user_engagement': {
                'avg_daily_activity': 8.5,
                'engagement_trend': 'STABLE',
                'active_user_percentage': 87.3,
                'engagement_consistency': 72.1
            },
            'risk_assessment': {
                'high_risk_users': 15,
                'medium_risk_users': 45,
                'low_risk_users': 940,
                'critical_alerts': 2,
                'overall_risk_level': 'MEDIUM'
            },
            'model_performance': {
                'best_model': 'Random Forest',
                'best_f1_score': 0.847,
                'best_accuracy': 0.923,
                'model_confidence': 84.2
            },
            'temporal_insights': {
                'users_declining': 23,
                'users_improving': 31,
                'stability_percentage': 85.6
            }
        }
        
        generator.summary_data['metrics'] = mock_metrics
        recommendations = generator.generate_recommendations()
        
        # Generate sample report
        html_content = generator._create_html_report(mock_metrics, recommendations)
        with open('sample_executive_report.html', 'w') as f:
            f.write(html_content)
        
        print("Sample executive report saved to sample_executive_report.html")
