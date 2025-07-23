# Coherence Protocol - Signal Mapping Prototype

## Overview
This project analyzes user behavior in terms of engagement and outputs a disengagement risk score based on various behavioral signals. It focuses on understanding potential signs of disengagement through message edits, meetings, and calendar voids.

## Files Included
- `signal_mapping.py`: The main script that processes the input data and calculates risk scores.
- `user_behavior_log.csv`: A sample CSV file containing mock data for users, including timestamps for message edits, meetings, and calendar voids.
- `README.md`: Documentation for the project.

## Data Schema
The CSV file must contain the following fields:
- `user_id`: Unique identifier for the user.
- `message_edit_timestamps`: Comma-separated timestamps when the user edited messages.
- `meeting_timestamps`: Comma-separated timestamps of user meetings.
- `calendar_voids`: Comma-separated date ranges with no scheduled meetings in the format `start:end`.

## Logic and Assumptions
- **Message Edits**: 
  - A gap of more than 5 days between edits signifies disengagement, contributing +20 to the risk score.
  
- **Meeting Frequency**: 
  - If there are no meetings occurring within a 14-day window, the user receives a +30 score.

- **Calendar Voids**:
  - A calendar void of more than 3 days contributes +40 to the risk score.

The maximum risk score is capped at 100.

## Bonus Component
The script includes a visualization with Matplotlib that displays each user’s disengagement risk score, with a threshold line at score 50.

## Requirements
- Python 3.x
- Pandas
- Matplotlib
- NumPy

## Usage
Run the script using the command: `python signal_mapping.py`

## Conclusion
This tool might serve as a prototype for evaluating user engagement in a multi-agent AI system aimed at detecting emotional and behavioral signals related to trust and alignment within teams