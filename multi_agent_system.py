"""
Multi-Agent Behavioral Signal Mapping System
Advanced disengagement detection with multiple specialized agents
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UserProfile:
    """User profile containing behavioral metrics"""
    user_id: str
    edit_patterns: Dict
    meeting_patterns: Dict
    calendar_patterns: Dict
    risk_scores: Dict
    timestamp: datetime
    
@dataclass 
class RiskAlert:
    """Risk alert structure"""
    user_id: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float
    contributing_factors: List[str]
    recommendations: List[str]
    timestamp: datetime
    agent_source: str

class BehavioralAgent(ABC):
    """Abstract base class for behavioral analysis agents"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.alerts = []
        
    @abstractmethod
    def analyze(self, user_data: pd.Series) -> Dict:
        """Analyze user behavior and return risk assessment"""
        pass
    
    @abstractmethod
    def get_risk_score(self, analysis: Dict) -> float:
        """Calculate risk score from analysis"""
        pass

class MessageEditAgent(BehavioralAgent):
    """Agent specialized in analyzing message editing patterns"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("MessageEditAgent", weight)
        self.edit_gap_threshold = 5  # days
        self.consistency_threshold = 3  # std dev
    
    def analyze(self, user_data: pd.Series) -> Dict:
        timestamps_str = user_data.get('message_edit_timestamps', '')
        
        if pd.isna(timestamps_str) or str(timestamps_str).strip() == '':
            return {
                'edit_count': 0,
                'max_gap': 0,
                'avg_gap': 0,
                'consistency': 0,
                'recent_activity': False,
                'risk_factors': []
            }
        
        try:
            times = [datetime.fromisoformat(t.strip()) for t in str(timestamps_str).split(",")]
            times.sort()
            
            if len(times) < 2:
                return {
                    'edit_count': len(times),
                    'max_gap': 0,
                    'avg_gap': 0,
                    'consistency': 0,
                    'recent_activity': len(times) > 0,
                    'risk_factors': ['Insufficient edit history'] if len(times) == 0 else []
                }
            
            gaps = [(times[i] - times[i-1]).days for i in range(1, len(times))]
            max_gap = max(gaps)
            avg_gap = np.mean(gaps)
            consistency = np.std(gaps)
            
            # Recent activity check (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_activity = any(t >= recent_cutoff for t in times)
            
            risk_factors = []
            if max_gap > self.edit_gap_threshold:
                risk_factors.append(f'Long edit gap: {max_gap} days')
            if consistency > self.consistency_threshold:
                risk_factors.append(f'Inconsistent editing: {consistency:.1f} std dev')
            if not recent_activity:
                risk_factors.append('No recent editing activity')
                
            return {
                'edit_count': len(times),
                'max_gap': max_gap,
                'avg_gap': avg_gap,
                'consistency': consistency,
                'recent_activity': recent_activity,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            return {
                'edit_count': 0,
                'max_gap': 0,
                'avg_gap': 0,
                'consistency': 0,
                'recent_activity': False,
                'risk_factors': [f'Parse error: {str(e)}']
            }
    
    def get_risk_score(self, analysis: Dict) -> float:
        score = 0.0
        
        # Gap-based scoring
        if analysis['max_gap'] > self.edit_gap_threshold:
            score += 0.4
        
        # Consistency scoring
        if analysis['consistency'] > self.consistency_threshold:
            score += 0.3
            
        # Recent activity scoring
        if not analysis['recent_activity']:
            score += 0.3
            
        return min(score, 1.0)

class MeetingPatternAgent(BehavioralAgent):
    """Agent specialized in analyzing meeting attendance patterns"""
    
    def __init__(self, weight: float = 0.30):
        super().__init__("MeetingPatternAgent", weight)
        self.min_meeting_frequency = 2  # per analysis period
        self.gap_threshold = 14  # days
    
    def analyze(self, user_data: pd.Series) -> Dict:
        timestamps_str = user_data.get('meeting_timestamps', '')
        
        if pd.isna(timestamps_str) or str(timestamps_str).strip() == '':
            return {
                'meeting_count': 0,
                'frequency_score': 0,
                'last_meeting_days': 999,
                'pattern_regularity': 0,
                'risk_factors': ['No meeting data']
            }
        
        try:
            times = [datetime.fromisoformat(t.strip()) for t in str(timestamps_str).split(",")]
            times.sort()
            
            if len(times) == 0:
                return {
                    'meeting_count': 0,
                    'frequency_score': 0,
                    'last_meeting_days': 999,
                    'pattern_regularity': 0,
                    'risk_factors': ['No meetings scheduled']
                }
            
            # Calculate metrics
            meeting_count = len(times)
            last_meeting_days = (datetime.now() - times[-1]).days
            
            if len(times) > 1:
                gaps = [(times[i] - times[i-1]).days for i in range(1, len(times))]
                pattern_regularity = 1 / (np.std(gaps) + 1)  # Higher = more regular
            else:
                pattern_regularity = 0
            
            frequency_score = min(meeting_count / self.min_meeting_frequency, 1.0)
            
            risk_factors = []
            if meeting_count < self.min_meeting_frequency:
                risk_factors.append(f'Low meeting frequency: {meeting_count}')
            if last_meeting_days > self.gap_threshold:
                risk_factors.append(f'Last meeting {last_meeting_days} days ago')
            if pattern_regularity < 0.5:
                risk_factors.append('Irregular meeting pattern')
                
            return {
                'meeting_count': meeting_count,
                'frequency_score': frequency_score,
                'last_meeting_days': last_meeting_days,
                'pattern_regularity': pattern_regularity,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            return {
                'meeting_count': 0,
                'frequency_score': 0,
                'last_meeting_days': 999,
                'pattern_regularity': 0,
                'risk_factors': [f'Parse error: {str(e)}']
            }
    
    def get_risk_score(self, analysis: Dict) -> float:
        score = 0.0
        
        # Frequency scoring
        score += max(0, (1 - analysis['frequency_score']) * 0.4)
        
        # Recent meeting scoring
        if analysis['last_meeting_days'] > self.gap_threshold:
            score += 0.4
            
        # Pattern regularity scoring
        if analysis['pattern_regularity'] < 0.3:
            score += 0.2
            
        return min(score, 1.0)

class CalendarVoidAgent(BehavioralAgent):
    """Agent specialized in analyzing calendar void patterns"""
    
    def __init__(self, weight: float = 0.25):
        super().__init__("CalendarVoidAgent", weight)
        self.void_threshold = 7  # days
    
    def analyze(self, user_data: pd.Series) -> Dict:
        voids_str = user_data.get('calendar_voids', '')
        
        if pd.isna(voids_str) or str(voids_str).strip() == '':
            return {
                'total_void_days': 0,
                'void_ratio': 0,
                'max_void_period': 0,
                'risk_factors': ['No calendar data']
            }
        
        try:
            void_str = str(voids_str).strip()
            parts = void_str.split(':')
            
            if len(parts) >= 6:
                start_date_str = ':'.join(parts[:3])
                end_date_str = ':'.join(parts[3:6])
                
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
                total_void_days = max((end_date - start_date).days, 0)
            else:
                total_void_days = 0
            
            # Calculate void ratio (simplified)
            analysis_period = 30  # days
            void_ratio = min(total_void_days / analysis_period, 1.0)
            
            risk_factors = []
            if total_void_days > self.void_threshold:
                risk_factors.append(f'Extended calendar void: {total_void_days} days')
            if void_ratio > 0.5:
                risk_factors.append(f'High void ratio: {void_ratio:.1%}')
                
            return {
                'total_void_days': total_void_days,
                'void_ratio': void_ratio,
                'max_void_period': total_void_days,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            return {
                'total_void_days': 0,
                'void_ratio': 0,
                'max_void_period': 0,
                'risk_factors': [f'Parse error: {str(e)}']
            }
    
    def get_risk_score(self, analysis: Dict) -> float:
        score = 0.0
        
        # Void duration scoring
        if analysis['total_void_days'] > self.void_threshold:
            score += 0.6
            
        # Void ratio scoring
        if analysis['void_ratio'] > 0.3:
            score += 0.4
            
        return min(score, 1.0)

class EngagementTrendAgent(BehavioralAgent):
    """Agent specialized in detecting engagement trends and patterns"""
    
    def __init__(self, weight: float = 0.20):
        super().__init__("EngagementTrendAgent", weight)
    
    def analyze(self, user_data: pd.Series) -> Dict:
        # Combine insights from other patterns
        edit_score = self._calculate_edit_trend(user_data.get('message_edit_timestamps', ''))
        meeting_score = self._calculate_meeting_trend(user_data.get('meeting_timestamps', ''))
        
        overall_trend = (edit_score + meeting_score) / 2
        
        risk_factors = []
        if edit_score > 0.5:
            risk_factors.append('Declining edit activity')
        if meeting_score > 0.5:
            risk_factors.append('Declining meeting participation')
        if overall_trend > 0.6:
            risk_factors.append('Overall engagement decline')
            
        return {
            'edit_trend_risk': edit_score,
            'meeting_trend_risk': meeting_score,
            'overall_trend_risk': overall_trend,
            'risk_factors': risk_factors
        }
    
    def _calculate_edit_trend(self, timestamps_str: str) -> float:
        if pd.isna(timestamps_str) or str(timestamps_str).strip() == '':
            return 0.5
        
        try:
            times = [datetime.fromisoformat(t.strip()) for t in str(timestamps_str).split(",")]
            if len(times) < 3:
                return 0.3
            
            times.sort()
            recent_times = [t for t in times if t >= datetime.now() - timedelta(days=14)]
            older_times = [t for t in times if t < datetime.now() - timedelta(days=14)]
            
            recent_rate = len(recent_times) / 14 if recent_times else 0
            older_rate = len(older_times) / 14 if older_times else 0
            
            if older_rate == 0:
                return 0.3
            
            decline_ratio = recent_rate / older_rate
            return max(0, 1 - decline_ratio)
            
        except:
            return 0.5
    
    def _calculate_meeting_trend(self, timestamps_str: str) -> float:
        # Similar logic for meeting trends
        if pd.isna(timestamps_str) or str(timestamps_str).strip() == '':
            return 0.5
        
        try:
            times = [datetime.fromisoformat(t.strip()) for t in str(timestamps_str).split(",")]
            if len(times) < 2:
                return 0.4
            
            times.sort()
            recent_meetings = len([t for t in times if t >= datetime.now() - timedelta(days=14)])
            
            if recent_meetings == 0:
                return 0.7
            elif recent_meetings < 2:
                return 0.5
            else:
                return 0.2
                
        except:
            return 0.5
    
    def get_risk_score(self, analysis: Dict) -> float:
        return analysis.get('overall_trend_risk', 0.5)

class MultiAgentDisengagementDetector:
    """Main orchestrator for multi-agent disengagement detection"""
    
    def __init__(self):
        self.agents = [
            MessageEditAgent(weight=0.25),
            MeetingPatternAgent(weight=0.30),
            CalendarVoidAgent(weight=0.25),
            EngagementTrendAgent(weight=0.20)
        ]
        self.alerts = []
        self.user_profiles = {}
    
    def analyze_user(self, user_data: pd.Series) -> Tuple[float, List[RiskAlert]]:
        """Analyze a single user with all agents"""
        user_id = user_data.get('user_id', 'Unknown')
        
        agent_results = {}
        total_risk = 0.0
        all_risk_factors = []
        
        # Run each agent
        for agent in self.agents:
            analysis = agent.analyze(user_data)
            risk_score = agent.get_risk_score(analysis)
            
            agent_results[agent.name] = {
                'analysis': analysis,
                'risk_score': risk_score,
                'weight': agent.weight
            }
            
            total_risk += risk_score * agent.weight
            all_risk_factors.extend(analysis.get('risk_factors', []))
        
        # Generate alerts based on risk level
        alerts = self._generate_alerts(user_id, total_risk, all_risk_factors, agent_results)
        
        # Create user profile
        profile = UserProfile(
            user_id=user_id,
            edit_patterns=agent_results.get('MessageEditAgent', {}).get('analysis', {}),
            meeting_patterns=agent_results.get('MeetingPatternAgent', {}).get('analysis', {}),
            calendar_patterns=agent_results.get('CalendarVoidAgent', {}).get('analysis', {}),
            risk_scores={name: res['risk_score'] for name, res in agent_results.items()},
            timestamp=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        
        return total_risk, alerts
    
    def _generate_alerts(self, user_id: str, risk_score: float, risk_factors: List[str], 
                        agent_results: Dict) -> List[RiskAlert]:
        """Generate appropriate alerts based on risk assessment"""
        alerts = []
        
        if risk_score >= 0.8:
            level = "CRITICAL"
            recommendations = [
                "Immediate intervention recommended",
                "Schedule one-on-one meeting",
                "Review workload and responsibilities",
                "Assess team dynamics and support needs"
            ]
        elif risk_score >= 0.6:
            level = "HIGH"
            recommendations = [
                "Monitor closely",
                "Increase check-in frequency",
                "Review recent project assignments",
                "Consider workload adjustment"
            ]
        elif risk_score >= 0.4:
            level = "MEDIUM"
            recommendations = [
                "Maintain regular monitoring",
                "Include in next team review",
                "Check for resource needs"
            ]
        else:
            level = "LOW"
            recommendations = ["Continue normal monitoring"]
        
        if level in ["MEDIUM", "HIGH", "CRITICAL"]:
            alert = RiskAlert(
                user_id=user_id,
                risk_level=level,
                risk_score=risk_score,
                contributing_factors=list(set(risk_factors)),
                recommendations=recommendations,
                timestamp=datetime.now(),
                agent_source="MultiAgentSystem"
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        return alerts
    
    def analyze_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze entire dataset"""
        results = []
        
        print(f"Analyzing {len(data)} users with multi-agent system...")
        
        for idx, user_data in data.iterrows():
            risk_score, alerts = self.analyze_user(user_data)
            
            results.append({
                'user_id': user_data.get('user_id', f'User_{idx}'),
                'risk_score': risk_score,
                'alert_count': len(alerts),
                'highest_alert_level': alerts[0].risk_level if alerts else 'LOW'
            })
        
        return pd.DataFrame(results)
    
    def get_summary_report(self) -> Dict:
        """Generate summary report"""
        if not self.alerts:
            return {"message": "No alerts generated"}
        
        alert_counts = {}
        for alert in self.alerts:
            alert_counts[alert.risk_level] = alert_counts.get(alert.risk_level, 0) + 1
        
        return {
            'total_alerts': len(self.alerts),
            'alert_breakdown': alert_counts,
            'high_risk_users': [alert.user_id for alert in self.alerts if alert.risk_level in ['HIGH', 'CRITICAL']],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def export_alerts(self, filename: str = 'disengagement_alerts.json'):
        """Export alerts to JSON file"""
        alert_data = []
        for alert in self.alerts:
            alert_data.append({
                'user_id': alert.user_id,
                'risk_level': alert.risk_level,
                'risk_score': alert.risk_score,
                'contributing_factors': alert.contributing_factors,
                'recommendations': alert.recommendations,
                'timestamp': alert.timestamp.isoformat(),
                'agent_source': alert.agent_source
            })
        
        with open(filename, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        print(f"Alerts exported to {filename}")

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("synthetic_user_behavior_log.csv")
    
    # Initialize multi-agent system
    detector = MultiAgentDisengagementDetector()
    
    # Analyze dataset
    results = detector.analyze_dataset(data)
    
    # Display results
    print("\nMulti-Agent Analysis Results:")
    print("=" * 50)
    print(f"Users analyzed: {len(results)}")
    print(f"Users with alerts: {len(results[results['alert_count'] > 0])}")
    
    risk_distribution = results['risk_score'].describe()
    print(f"\nRisk Score Distribution:")
    print(f"Mean: {risk_distribution['mean']:.3f}")
    print(f"Std:  {risk_distribution['std']:.3f}")
    print(f"Max:  {risk_distribution['max']:.3f}")
    
    # Show high-risk users
    high_risk = results[results['risk_score'] > 0.6].sort_values('risk_score', ascending=False)
    if len(high_risk) > 0:
        print(f"\nTop 10 High-Risk Users:")
        print(high_risk.head(10).to_string(index=False))
    
    # Generate summary report
    summary = detector.get_summary_report()
    print(f"\nSummary Report:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export alerts
    detector.export_alerts()
