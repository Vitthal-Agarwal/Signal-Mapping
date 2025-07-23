import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Constants for scoring
EDIT_GAP_THRESHOLD = 5  # days
MEETING_GAP_THRESHOLD = 14  # days
CALENDAR_VOID_THRESHOLD = 3  # days

def calculate_risk_score(user_data):
    current_score = 0
    
    # Analyzing message edits
    edit_times = [
        datetime.fromisoformat(ts.strip()) 
        for ts in user_data['message_edit_timestamps'].split(',')
    ]
    edit_gaps = np.diff(sorted(edit_times)).astype('timedelta64[D]').astype(int)
    long_edit_gaps = edit_gaps[edit_gaps > EDIT_GAP_THRESHOLD]
    current_score += len(long_edit_gaps) * 20

    # Analyzing meetings
    meeting_times = [
        datetime.fromisoformat(ts.strip()) 
        for ts in user_data['meeting_timestamps'].split(',')
    ]
    if len(meeting_times) > 1:
        meeting_gaps = np.diff(sorted(meeting_times)).astype('timedelta64[D]').astype(int)
        if (meeting_gaps[-1] > MEETING_GAP_THRESHOLD):
            current_score += 30

    # Analyzing calendar voids
    voids = [
        date_range.split(':') 
        for date_range in user_data['calendar_voids'].split(',')
    ]
    for start, end in voids:
        start_date = datetime.fromisoformat(start.strip())
        end_date = datetime.fromisoformat(end.strip())
        days_void = (end_date - start_date).days
        if days_void > CALENDAR_VOID_THRESHOLD:
            current_score += 40

    return min(current_score, 100)  # cap at 100

def main():
    # Load data
    df = pd.read_csv('user_behavior_log.csv')
    
    # Analyze risk scores
    df['risk_score'] = df.apply(calculate_risk_score, axis=1)

    # Output the scores
    print(df[['user_id', 'risk_score']])

    # Bonus: Visualize the scores
    plt.figure(figsize=(10, 6))
    plt.bar(df['user_id'].astype(str), df['risk_score'], color='royalblue')
    plt.ylim(0, 100)
    plt.xlabel('User ID')
    plt.ylabel('Disengagement Risk Score')
    plt.title('Disengagement Risk Scores by User')
    plt.axhline(y=50, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()