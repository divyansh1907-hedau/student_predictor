import io
import json
import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from flask_mail import Mail, Message
from google import genai
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'edupredict-enterprise-secret-key-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///edupredict.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ==========================================
# 1. CONFIGURATION & SDK INITIALIZATION
# ==========================================

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'edupredict.ai@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
app.config['MAIL_DEFAULT_SENDER'] = ('EduPredict AI Advisory', 'edupredict.ai@gmail.com')

mail = Mail(app)

# Initialize Google Gemini AI Client safely from Environment Variables
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# Load ML artifacts safely
model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None

# Global variable to store last batch results for PDF export and email dispatch
last_batch_results = []

metrics_summary = {}
if os.path.exists('metrics.json'):
    with open('metrics.json', 'r') as f:
        metrics_summary = json.load(f)


# ==========================================
# 2. DATABASE MODELS (RBAC, Multi-Subject Grades)
# ==========================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin', 'faculty', 'student'
    subject = db.Column(db.String(80), nullable=True, default='General')  # Subject assigned to Faculty
    
    # Relationship to per-subject grades (for student role)
    grades = db.relationship('StudentSubjectGrade', backref='student_user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class StudentSubjectGrade(db.Model):
    """Stores attendance and marks per student per subject."""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(80), nullable=False)
    
    attendance = db.Column(db.Float, default=75.0)
    midterm = db.Column(db.Float, default=60.0)
    assignment = db.Column(db.Float, default=60.0)
    logins = db.Column(db.Float, default=12.0)
    study_hours = db.Column(db.Float, default=5.0)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ==========================================
# 3. HELPER & INITIALIZATION FUNCTIONS
# ==========================================

def init_db():
    """Creates tables and seeds default multi-subject faculty accounts and student subject records."""
    db.create_all()

    # Seed Admin
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@university.edu', role='admin', subject='Administration')
        admin.set_password('admin123')
        db.session.add(admin)

    # Seed Subject Faculties
    subject_faculties = [
        {'username': 'faculty_ds', 'email': 'ds_faculty@university.edu', 'subject': 'Data Structures & Algorithms'},
        {'username': 'faculty_dbms', 'email': 'dbms_faculty@university.edu', 'subject': 'Database Management Systems'},
        {'username': 'faculty_os', 'email': 'os_faculty@university.edu', 'subject': 'Operating Systems'},
        {'username': 'faculty_ai', 'email': 'ai_faculty@university.edu', 'subject': 'Artificial Intelligence'}
    ]

    for f_data in subject_faculties:
        if not User.query.filter_by(username=f_data['username']).first():
            f_user = User(
                username=f_data['username'],
                email=f_data['email'],
                role='faculty',
                subject=f_data['subject']
            )
            f_user.set_password('faculty123')
            db.session.add(f_user)

    # Seed Demo Student with records in 2 subjects
    student = User.query.filter_by(username='student1').first()
    if not student:
        student = User(username='student1', email='student1@university.edu', role='student')
        student.set_password('student123')
        db.session.add(student)
        db.session.flush()

        # Add sample subject grades for student1
        grade_ds = StudentSubjectGrade(
            student_id=student.id, subject='Data Structures & Algorithms',
            attendance=85.0, midterm=78.0, assignment=80.0, logins=15.0, study_hours=6.0
        )
        grade_dbms = StudentSubjectGrade(
            student_id=student.id, subject='Database Management Systems',
            attendance=48.0, midterm=38.0, assignment=42.0, logins=6.0, study_hours=2.0
        )
        db.session.add_all([grade_ds, grade_dbms])

    db.session.commit()

with app.app_context():
    init_db()


def generate_personalized_path(midterm, assignment, attendance, study_hours):
    """Generates dynamic remediation recommendations based on student performance metrics."""
    path = []
    if midterm < 50:
        if study_hours < 5:
            path.append({
                'phase': 'Week 1-2: Concept Rebuilding',
                'recommendation': 'Low study time detected. Engage in 15-minute daily micro-learning modules.',
                'type': 'Academic'
            })
        else:
            path.append({
                'phase': 'Week 1-2: Study Strategy Pivot',
                'recommendation': 'High effort but low returns. Transition from passive reading to active recall & practice testing.',
                'type': 'Methodology'
            })

    if assignment < 50:
        path.append({
            'phase': 'Week 2: Applied Practice',
            'recommendation': 'Complete guided problem sets with peer-mentoring assistance.',
            'type': 'Practical'
        })

    if attendance < 75:
        path.append({
            'phase': 'Week 3: Attendance & Routine Alignment',
            'recommendation': 'Schedule mandatory counseling check-ins and set automated lecture alerts.',
            'type': 'Habit'
        })

    if not path:
        path.append({
            'phase': 'Maintenance & Enrichment',
            'recommendation': 'Student performing well. Assign advanced peer-tutoring or project extension tasks.',
            'type': 'Enrichment'
        })

    path.append({
        'phase': 'Week 4: Mastery Verification',
        'recommendation': 'Attempt a simulated diagnostic mock assessment to re-evaluate risk profile.',
        'type': 'Assessment'
    })

    return path


# ==========================================
# 4. AUTHENTICATION & PORTAL ROUTES
# ==========================================

@app.route('/')
def home():
    """Root route: Forces unauthenticated users to login, or redirects to their role dashboard."""
    if not current_user.is_authenticated:
        return redirect(url_for('login'))

    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'student':
        return redirect(url_for('student_dashboard'))
    else:
        return render_template('index.html', user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'student':
                return redirect(url_for('student_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    """HOD / Admin Executive Dashboard."""
    if current_user.role != 'admin':
        return "Unauthorized Access", 403
    total_users = User.query.count()
    students = User.query.filter_by(role='student').all()
    faculty = User.query.filter_by(role='faculty').all()
    return render_template('admin.html', total_users=total_users, students=students, faculty=faculty)


@app.route('/student_dashboard')
@login_required
def student_dashboard():
    """Personalized Student Portal with Per-Subject Breakdowns."""
    if current_user.role != 'student':
        return "Unauthorized Access", 403
    
    subject_cards = []
    total_risk = 0.0

    for g in current_user.grades:
        risk_val = 25.0
        at_risk = False
        if model and scaler:
            raw_features = np.array([[g.attendance, g.midterm, g.assignment, g.logins, g.study_hours]])
            scaled_features = scaler.transform(raw_features)
            prediction = model.predict(scaled_features)[0]
            prob = model.predict_proba(scaled_features)[0][1] if hasattr(model, 'predict_proba') else float(prediction)
            risk_val = round(prob * 100, 2)
            at_risk = bool(prediction == 1 or risk_val >= 50)

        remediation = generate_personalized_path(g.midterm, g.assignment, g.attendance, g.study_hours)
        total_risk += risk_val

        subject_cards.append({
            'subject': g.subject,
            'attendance': g.attendance,
            'midterm': g.midterm,
            'assignment': g.assignment,
            'study_hours': g.study_hours,
            'risk_percent': risk_val,
            'at_risk': at_risk,
            'remediation': remediation
        })

    overall_risk = round(total_risk / len(subject_cards), 2) if subject_cards else 25.0
    overall_at_risk = bool(overall_risk >= 50)
    remediation_path = subject_cards[0]['remediation'] if subject_cards else generate_personalized_path(60, 60, 75, 5)

    return render_template(
        'student.html', 
        student=current_user, 
        subject_cards=subject_cards,
        risk_percent=overall_risk,
        at_risk=overall_at_risk,
        remediation_path=remediation_path
    )


# ==========================================
# 5. PER-SUBJECT CSV BATCH PROCESSING
# ==========================================

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics_summary)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

    try:
        attendance = float(request.form.get('attendance', 0))
        midterm = float(request.form.get('midterm', 0))
        assignment = float(request.form.get('assignment', 0))
        logins = float(request.form.get('logins', 0))
        study_hours = float(request.form.get('study_hours', 0))

        raw_features = np.array([[attendance, midterm, assignment, logins, study_hours]])
        scaled_features = scaler.transform(raw_features)

        prediction = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0][1] if hasattr(model, 'predict_proba') else float(prediction)

        risk_percent = round(prob * 100, 2)
        at_risk = bool(prediction == 1)

        remediation_path = generate_personalized_path(midterm, assignment, attendance, study_hours)

        return render_template(
            'index.html',
            user=current_user,
            prediction_text=f'Student Failure Risk: {risk_percent}%',
            at_risk=at_risk,
            risk_percent=risk_percent,
            remediation_path=remediation_path,
            midterm=midterm,
            assignment=assignment,
            study_hours=study_hours
        )
    except Exception as e:
        return render_template('index.html', user=current_user, prediction_text=f'Error making prediction: {str(e)}')


@app.route('/upload_batch', methods=['POST'])
@login_required
def upload_batch():
    """Batch CSV processing updating subject-specific student records."""
    global last_batch_results
    try:
        file = request.files.get('file')
        if not file or not file.filename.endswith('.csv'):
            return render_template('index.html', user=current_user, batch_error='Please upload a valid CSV file.')

        df = pd.read_csv(file)
        required_cols = ['attendance', 'midterm', 'assignment', 'logins', 'study_hours']

        df.columns = [c.lower().strip() for c in df.columns]

        for col in required_cols:
            if col not in df.columns:
                return render_template('index.html', user=current_user, batch_error=f'Missing column: {col}')

        # 1. ML Batch Prediction
        X = df[required_cols]
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        probabilities = (
            model.predict_proba(X_scaled)[:, 1]
            if hasattr(model, 'predict_proba')
            else predictions
        )

        faculty_subject = getattr(current_user, 'subject', 'General')
        existing_users = {u.username: u for u in User.query.all()}
        default_password_hash = generate_password_hash('student123')

        batch_results = []
        new_users_to_add = []

        for idx, row in df.iterrows():
            risk_val = round(probabilities[idx] * 100, 1)
            raw_name = str(row.get('name', f'student_{idx+1}')).strip()
            student_username = raw_name.replace(' ', '_').lower()
            student_email = row.get('email', f'{student_username}@university.edu')
            student_id = row.get('id', f'STU-{idx+1000}')

            # Find or Create Student User
            if student_username in existing_users:
                student_user = existing_users[student_username]
            else:
                student_user = User(
                    username=student_username,
                    email=student_email,
                    role='student',
                    password_hash=default_password_hash
                )
                new_users_to_add.append(student_user)
                existing_users[student_username] = student_user

            # Update or Create Subject Grade Entry for this Faculty's Subject
            grade_entry = StudentSubjectGrade.query.filter_by(
                student_id=student_user.id, subject=faculty_subject
            ).first() if student_user.id else None

            if not grade_entry:
                grade_entry = StudentSubjectGrade(
                    student_user=student_user,
                    subject=faculty_subject,
                    attendance=float(row['attendance']),
                    midterm=float(row['midterm']),
                    assignment=float(row['assignment']),
                    logins=float(row['logins']),
                    study_hours=float(row['study_hours'])
                )
                db.session.add(grade_entry)
            else:
                grade_entry.attendance = float(row['attendance'])
                grade_entry.midterm = float(row['midterm'])
                grade_entry.assignment = float(row['assignment'])
                grade_entry.logins = float(row['logins'])
                grade_entry.study_hours = float(row['study_hours'])

            batch_results.append({
                'id': student_id,
                'name': raw_name,
                'email': student_email,
                'subject': faculty_subject,
                'attendance': row['attendance'],
                'midterm': row['midterm'],
                'assignment': row['assignment'],
                'logins': row['logins'],
                'study_hours': row['study_hours'],
                'risk_percent': risk_val,
                'at_risk': bool(predictions[idx] == 1 or risk_val >= 50)
            })

        if new_users_to_add:
            db.session.add_all(new_users_to_add)
        db.session.commit()

        last_batch_results = batch_results
        at_risk_count = sum(1 for s in batch_results if s['at_risk'])

        flash(f'Batch CSV processed for [{faculty_subject}]! Updated {len(batch_results)} student records ({len(new_users_to_add)} new logins created).', 'success')

        return render_template(
            'index.html',
            user=current_user,
            batch_results=batch_results,
            total_students=len(batch_results),
            at_risk_count=at_risk_count
        )
    except Exception as e:
        return render_template('index.html', user=current_user, batch_error=f'Error processing CSV: {str(e)}')


# ==========================================
# 6. OTHER ROUTES (Mail, AI Practice, PDF Reports)
# ==========================================

@app.route('/send_intervention', methods=['POST'])
@login_required
def send_intervention():
    try:
        data = request.json or {}
        student_id = data.get('student_id')

        student = next((s for s in last_batch_results if str(s['id']) == str(student_id)), None)
        if not student:
            return jsonify({'success': False, 'message': 'Student record not found.'}), 404

        path = generate_personalized_path(
            student['midterm'], student['assignment'], student['attendance'], student['study_hours']
        )

        remediation_text = '\n'.join([f"• {item['phase']}: {item['recommendation']}" for item in path])

        email_body = f"""Dear {student['name']},

This is an automated academic advisory alert from the EduPredict AI platform regarding [{student.get('subject', 'Course')}].

Performance Metrics:
- Subject: {student.get('subject', 'General')}
- Calculated Failure Risk Score: {student['risk_percent']}%
- Attendance: {student['attendance']}%
- Midterm Score: {student['midterm']}%

Recommended 4-Week Remediation Roadmap:
{remediation_text}

Please log into your EduPredict Student Portal to complete your subject-specific AI practice modules.

Best regards,
EduPredict Academic Support Team
"""

        try:
            msg = Message(
                subject=f"⚠️ Academic Intervention Notice ({student.get('subject', 'Course')}): {student['name']}",
                recipients=[student['email']],
                body=email_body
            )
            mail.send(msg)
            status_msg = f"Intervention email successfully sent to {student['email']}!"
        except Exception:
            status_msg = f"Intervention alert logged for {student['name']} ({student['email']})!"

        return jsonify({'success': True, 'message': status_msg})

    except Exception as e:
        return jsonify({'success': False, 'message': f"Error: {str(e)}"}), 500


@app.route('/generate_practice', methods=['POST'])
@login_required
def generate_practice():
    try:
        data = request.json or {}
        midterm = float(data.get('midterm', 50))
        assignment = float(data.get('assignment', 50))
        study_hours = float(data.get('study_hours', 5))

        focus_area = []
        if midterm < 50:
            focus_area.append("Core Conceptual Theory & Fundamentals")
        if assignment < 50:
            focus_area.append("Practical Problem Solving & Code Analysis")
        if study_hours < 5:
            focus_area.append("High-Yield Quick Recall Questions")

        if not focus_area:
            focus_area.append("Advanced Logic & Computer Science Fundamentals")

        fallback_questions = [
            {
                "id": 1,
                "topic": "Algorithm Time Complexity",
                "question": "What is the average time complexity of searching an element in a balanced Binary Search Tree (BST)?",
                "options": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n log n)"],
                "correct_answer": "B) O(log n)",
                "explanation": "In a balanced BST, each comparison eliminates half of the remaining elements, resulting in logarithmic O(log n) performance."
            },
            {
                "id": 2,
                "topic": "Operating Systems & Deadlocks",
                "question": "Which of the following is NOT one of Coffman’s four necessary conditions for a deadlock to occur?",
                "options": ["A) Mutual Exclusion", "B) Hold and Wait", "C) Preemption Allowed", "D) Circular Wait"],
                "correct_answer": "C) Preemption Allowed",
                "explanation": "The required condition is 'No Preemption'. If preemption is allowed, resources can be reclaimed, preventing deadlocks."
            },
            {
                "id": 3,
                "topic": "Database Normalization",
                "question": "A relational database table is in 2NF (Second Normal Form) if it is in 1NF and:",
                "options": ["A) Has no transitive dependencies", "B) Has no partial key dependencies", "C) All attributes are multi-valued", "D) Contains no foreign keys"],
                "correct_answer": "B) Has no partial key dependencies",
                "explanation": "2NF requires that all non-key attributes depend fully on the entire primary key, eliminating partial functional dependencies."
            }
        ]

        if ai_client:
            prompt = f"""
You are an expert AI academic tutor. A university student has the following academic metrics:
- Midterm Score: {midterm}%
- Assignment Score: {assignment}%
- Weekly Study Hours: {study_hours} hrs/week
- Focus Areas Needed: {', '.join(focus_area)}

Generate 3 multiple-choice practice questions specifically tailored to help this student improve their weak points in Computer Science and Information Technology.

Return the response STRICTLY as a valid JSON array of objects with the following format:
[
  {{
    "id": 1,
    "topic": "Topic Name",
    "question": "Clear question text?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A) Option 1",
    "explanation": "Brief explanation of why this answer is correct."
  }}
]
Do not wrap response in markdown blocks like ```json. Return raw JSON text only.
"""
            model_options = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']
            response = None

            for m in model_options:
                try:
                    response = ai_client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    if response and response.text:
                        break
                except Exception:
                    continue

            if response and response.text:
                clean_text = response.text.replace("```json", "").replace("```", "").strip()
                questions = json.loads(clean_text)
                return jsonify({'success': True, 'questions': questions})

        return jsonify({'success': True, 'questions': fallback_questions})

    except Exception:
        return jsonify({'success': True, 'questions': fallback_questions})


@app.route('/export_report', methods=['GET'])
@login_required
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

    faculty_subject = getattr(current_user, 'subject', 'General')
    elements.append(Paragraph(f'🎓 EduPredict AI - Academic At-Risk Report ({faculty_subject})', title_style))
    at_risk_list = [s for s in last_batch_results if s['at_risk']]
    elements.append(Paragraph(f'Generated Batch Summary • Total Students Analyzed: {len(last_batch_results)} | Flagged At-Risk: {len(at_risk_list)}', subtitle_style))

    table_data = [['Student ID', 'Name', 'Subject', 'Attendance %', 'Midterm %', 'Risk Score']]
    for s in last_batch_results:
        table_data.append([
            str(s['id']),
            str(s['name']),
            str(s.get('subject', faculty_subject)),
            f"{s['attendance']}%",
            f"{s['midterm']}%",
            f"{s['risk_percent']}%" + (" ⚠️" if s['at_risk'] else " ✅")
        ])

    t = Table(table_data, colWidths=[70, 120, 110, 70, 70, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))

    elements.append(t)
    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name=f'EduPredict_{faculty_subject.replace(" ", "_")}_Report.pdf', mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)