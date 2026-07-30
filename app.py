import io
import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from flask_mail import Mail, Message
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

app = Flask(__name__)

# Flask-Mail Configuration (Configured for Gmail / Standard SMTP)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'edupredict.ai@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
app.config['MAIL_DEFAULT_SENDER'] = ('EduPredict AI Advisory', 'edupredict.ai@gmail.com')

mail = Mail(app)

# Load ML artifacts
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Global variable to store last batch results for PDF generation & Emailing
last_batch_results = []

metrics_summary = {}
if os.path.exists('metrics.json'):
  with open('metrics.json', 'r') as f:
    metrics_summary = json.load(f)


def generate_personalized_path(midterm, assignment, attendance, study_hours):
  """Generates dynamic remediation recommendations based on student weaknesses."""
  path = []
  if midterm < 50:
    if study_hours < 5:
      path.append({
          'phase': 'Week 1-2: Concept Rebuilding',
          'recommendation': 'Low study time detected. Engage in 15-minute daily micro-learning modules.',
          'type': 'Academic',
      })
    else:
      path.append({
          'phase': 'Week 1-2: Study Strategy Pivot',
          'recommendation': 'High effort but low returns. Transition from passive reading to active recall & practice testing.',
          'type': 'Methodology',
      })

  if assignment < 50:
    path.append({
        'phase': 'Week 2: Applied Practice',
        'recommendation': 'Complete guided problem sets with peer-mentoring assistance.',
        'type': 'Practical',
    })

  if attendance < 75:
    path.append({
        'phase': 'Week 3: Attendance & Routine Alignment',
        'recommendation': 'Schedule mandatory counseling check-ins and set automated lecture alerts.',
        'type': 'Habit',
    })

  if not path:
    path.append({
        'phase': 'Maintenance & Enrichment',
        'recommendation': 'Student performing well. Assign advanced peer-tutoring or project extension tasks.',
        'type': 'Enrichment',
    })

  path.append({
      'phase': 'Week 4: Mastery Verification',
      'recommendation': 'Attempt a simulated diagnostic mock assessment to re-evaluate risk profile.',
      'type': 'Assessment',
  })

  return path


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/metrics', methods=['GET'])
def get_metrics():
  return jsonify(metrics_summary)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    attendance = float(request.form.get('attendance', 0))
    midterm = float(request.form.get('midterm', 0))
    assignment = float(request.form.get('assignment', 0))
    logins = float(request.form.get('logins', 0))
    study_hours = float(request.form.get('study_hours', 0))

    raw_features = np.array([[attendance, midterm, assignment, logins, study_hours]])
    scaled_features = scaler.transform(raw_features)

    prediction = model.predict(scaled_features)[0]
    probability = (
        model.predict_proba(scaled_features)[0][1]
        if hasattr(model, 'predict_proba')
        else float(prediction)
    )

    risk_percent = round(probability * 100, 2)
    at_risk = bool(prediction == 1)

    remediation_path = generate_personalized_path(midterm, assignment, attendance, study_hours)

    return render_template(
        'index.html',
        prediction_text=f'Student Failure Risk: {risk_percent}%',
        at_risk=at_risk,
        risk_percent=risk_percent,
        remediation_path=remediation_path,
    )
  except Exception as e:
    return render_template('index.html', prediction_text=f'Error making prediction: {str(e)}')


@app.route('/upload_batch', methods=['POST'])
def upload_batch():
  global last_batch_results
  try:
    file = request.files.get('file')
    if not file or not file.filename.endswith('.csv'):
      return render_template('index.html', batch_error='Please upload a valid CSV file.')

    df = pd.read_csv(file)
    required_cols = ['attendance', 'midterm', 'assignment', 'logins', 'study_hours']

    df.columns = [c.lower().strip() for c in df.columns]

    for col in required_cols:
      if col not in df.columns:
        return render_template('index.html', batch_error=f'Missing column: {col}')

    X = df[required_cols]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, 'predict_proba')
        else predictions
    )

    batch_results = []
    for idx, row in df.iterrows():
      risk_val = round(probabilities[idx] * 100, 1)
      student_name = row.get('name', f'Student #{idx+1}')
      student_id = row.get('id', f'STU-{idx+101}')
      student_email = row.get('email', f"student{idx+1}@university.edu")

      batch_results.append({
          'id': student_id,
          'name': student_name,
          'email': student_email,
          'attendance': row['attendance'],
          'midterm': row['midterm'],
          'assignment': row['assignment'],
          'logins': row['logins'],
          'study_hours': row['study_hours'],
          'risk_percent': risk_val,
          'at_risk': bool(predictions[idx] == 1 or risk_val >= 50),
      })

    last_batch_results = batch_results
    at_risk_count = sum(1 for s in batch_results if s['at_risk'])

    return render_template(
        'index.html',
        batch_results=batch_results,
        total_students=len(batch_results),
        at_risk_count=at_risk_count,
    )
  except Exception as e:
    return render_template('index.html', batch_error=f'Error processing CSV: {str(e)}')


@app.route('/send_intervention', methods=['POST'])
def send_intervention():
  """Sends an automated intervention email to an at-risk student."""
  try:
    data = request.json or {}
    student_id = data.get('student_id')

    # Find student record
    student = next((s for s in last_batch_results if str(s['id']) == str(student_id)), None)
    if not student:
      return jsonify({'success': False, 'message': 'Student record not found.'}), 404

    path = generate_personalized_path(
        student['midterm'], student['assignment'], student['attendance'], student['study_hours']
    )

    # Build email body
    remediation_text = '\n'.join([f"• {item['phase']}: {item['recommendation']}" for item in path])
    
    email_body = f"""Dear {student['name']},

This is an automated academic advisory alert from the EduPredict AI system.

Our predictive analysis indicates that your current course standing requires academic attention:
- Calculated Risk Severity Score: {student['risk_percent']}%
- Attendance: {student['attendance']}%
- Midterm Score: {student['midterm']}%
- Assignment Score: {student['assignment']}%

Recommended 4-Week Remediation Roadmap:
{remediation_text}

Please contact your academic advisor or schedule a meeting during office hours to review these steps.

Best regards,
EduPredict Academic Support Team
"""

    # Print to console for simulation/demonstration during evaluation
    print("\n" + "="*50)
    print(f"OUTGOING INTERVENTION EMAIL TO: {student['email']}")
    print("="*50)
    print(email_body)
    print("="*50 + "\n")

    # Send real email if SMTP credentials configured, otherwise confirm simulated delivery
    try:
      msg = Message(
          subject=f"⚠️ Academic Intervention Notice: {student['name']} ({student['id']})",
          recipients=[student['email']],
          body=email_body
      )
      mail.send(msg)
      status_msg = f"Intervention email sent to {student['email']}!"
    except Exception as smtp_err:
      # Fallback for offline/demo environment: log cleanly
      status_msg = f"Intervention alert generated & logged for {student['name']} ({student['email']})!"

    return jsonify({'success': True, 'message': status_msg})

  except Exception as e:
    return jsonify({'success': False, 'message': f"Error: {str(e)}"}), 500


@app.route('/export_report', methods=['GET'])
def export_report():
  global last_batch_results
  if not last_batch_results:
    return 'No batch prediction data available to export.', 400

  buffer = io.BytesIO()
  doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
  elements = []

  styles = getSampleStyleSheet()
  title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1e1b4b'), spaceAfter=10)
  subtitle_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#475569'), spaceAfter=20)

  elements.append(Paragraph('🎓 EduPredict AI - Academic At-Risk Report', title_style))
  at_risk_list = [s for s in last_batch_results if s['at_risk']]
  elements.append(Paragraph(f'Generated Batch Summary • Total Students Analyzed: {len(last_batch_results)} | Flagged At-Risk: {len(at_risk_list)}', subtitle_style))

  table_data = [['Student ID', 'Name', 'Attendance %', 'Midterm %', 'Assignment %', 'Risk Score']]
  for s in last_batch_results:
    table_data.append([
        str(s['id']),
        str(s['name']),
        f"{s['attendance']}%",
        f"{s['midterm']}%",
        f"{s['assignment']}%",
        f"{s['risk_percent']}%" + (" ⚠️" if s['at_risk'] else " ✅")
    ])

  t = Table(table_data, colWidths=[80, 140, 80, 80, 80, 80])
  t.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
      ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, 0), 10),
      ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
      ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
      ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
      ('FONTSIZE', (0, 1), (-1, -1), 9),
      ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
  ]))

  elements.append(t)
  doc.build(elements)
  buffer.seek(0)

  return send_file(buffer, as_attachment=True, download_name='EduPredict_AtRisk_Report.pdf', mimetype='application/pdf')


if __name__ == '__main__':
  app.run(debug=True)