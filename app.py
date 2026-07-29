from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained best model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    attendance = float(data['attendance'])
    midterm = float(data['midterm'])
    assignment = float(data['assignment'])
    logins = float(data['logins'])
    study_hours = float(data.get('study_hours', 5))
    
    features = np.array([[attendance, midterm, assignment, logins, study_hours]])
    prediction = model.predict(features)[0]
    
    probabilities = model.predict_proba(features)[0]
    risk_percentage = int(round(probabilities[1] * 100)) if len(probabilities) > 1 else (80 if prediction == 1 else 20)
    
    # Early Intervention Recommendations (PPT Slide 13)
    recommendations = []
    if attendance < 75:
        recommendations.append(f"Low Attendance ({attendance}%): Issue automated attendance alert & schedule counseling.")
    if midterm < 50:
        recommendations.append(f"Low Midterm Mark ({midterm}): Recommend targeted peer tutoring modules.")
    if assignment < 50:
        recommendations.append(f"Low Assignment Score ({assignment}%): Send remedial learning assignment link.")
    if logins < 10:
        recommendations.append(f"Low LMS Portal Logins ({logins}): Trigger automated portal engagement notification.")
    if study_hours < 5:
        recommendations.append(f"Low Weekly Study Hours ({study_hours} hrs): Recommend time-management workshop.")

    return jsonify({
        'is_at_risk': bool(prediction == 1),
        'risk_percentage': risk_percentage,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    print("Starting PPT-Compliant EduPredict Server...")
    app.run(debug=True, port=5000)