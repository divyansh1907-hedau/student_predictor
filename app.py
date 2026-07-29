import json
import os
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load ML artifacts
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load metrics summary if available
metrics_summary = {}
if os.path.exists('metrics.json'):
    with open('metrics.json', 'r') as f:
        metrics_summary = json.load(f)


def generate_personalized_path(midterm, assignment, attendance, study_hours):
    """Generates dynamic remediation recommendations based on student weaknesses."""
    path = []

    # Academic Performance Thresholds
    if midterm < 50:
        if study_hours < 5:
            path.append({
                'phase': 'Week 1-2: Concept Rebuilding',
                'recommendation': (
                    'Low study time detected. Engage in 15-minute daily'
                    ' micro-learning modules.'
                ),
                'type': 'Academic',
            })
        else:
            path.append({
                'phase': 'Week 1-2: Study Strategy Pivot',
                'recommendation': (
                    'High effort but low returns. Transition from passive'
                    ' reading to active recall & practice testing.'
                ),
                'type': 'Methodology',
            })

    if assignment < 50:
        path.append({
            'phase': 'Week 2: Applied Practice',
            'recommendation': (
                'Complete guided problem sets with peer-mentoring assistance.'
            ),
            'type': 'Practical',
        })

    # Engagement & Habit Thresholds
    if attendance < 75:
        path.append({
            'phase': 'Week 3: Attendance & Routine Alignment',
            'recommendation': (
                'Schedule mandatory counseling check-ins and set automated'
                ' lecture alerts.'
            ),
            'type': 'Habit',
        })

    # Default baseline recommendation if performance is good
    if not path:
        path.append({
            'phase': 'Maintenance & Enrichment',
            'recommendation': (
                'Student performing well. Assign advanced peer-tutoring or'
                ' project extension tasks.'
            ),
            'type': 'Enrichment',
        })

    path.append({
        'phase': 'Week 4: Mastery Verification',
        'recommendation': (
            'Attempt a simulated diagnostic mock assessment to re-evaluate'
            ' risk profile.'
        ),
        'type': 'Assessment',
    })

    return path


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Endpoint for frontend charts to fetch algorithm performance comparison."""
    return jsonify(metrics_summary)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        attendance = float(request.form.get('attendance', 0))
        midterm = float(request.form.get('midterm', 0))
        assignment = float(request.form.get('assignment', 0))
        logins = float(request.form.get('logins', 0))
        study_hours = float(request.form.get('study_hours', 0))

        # Prepare raw array and apply standard scaler
        raw_features = np.array(
            [[attendance, midterm, assignment, logins, study_hours]]
        )
        scaled_features = scaler.transform(raw_features)

        # Make prediction & obtain risk probability
        prediction = model.predict(scaled_features)[0]
        probability = (
            model.predict_proba(scaled_features)[0][1]
            if hasattr(model, 'predict_proba')
            else float(prediction)
        )

        risk_percent = round(probability * 100, 2)
        at_risk = bool(prediction == 1)

        # Generate dynamic intervention path
        remediation_path = generate_personalized_path(
            midterm, assignment, attendance, study_hours
        )

        return render_template(
            'index.html',
            prediction_text=f'Student Failure Risk: {risk_percent}%',
            at_risk=at_risk,
            risk_percent=risk_percent,
            remediation_path=remediation_path,
        )
    except Exception as e:
        return render_template(
            'index.html', prediction_text=f'Error making prediction: {str(e)}'
        )


if __name__ == '__main__':
    app.run(debug=True)