import io
import json
import os
import time
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from flask_mail import Mail, Message
from google import genai

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'edupredict-enterprise-secret-key-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///edupredict.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Config video uploads
UPLOAD_FOLDER = os.path.join('static', 'uploads', 'videos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None

# Global store for batch results report generation
last_batch_results = []

# ==========================================
# 2. DATABASE MODELS
# ==========================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    subject = db.Column(db.String(80), nullable=True, default='General')
    
    grades = db.relationship('StudentSubjectGrade', backref='student_user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class StudentSubjectGrade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(80), nullable=False)
    
    attendance = db.Column(db.Float, default=75.0)
    midterm = db.Column(db.Float, default=60.0)
    assignment = db.Column(db.Float, default=60.0)
    logins = db.Column(db.Float, default=12.0)
    study_hours = db.Column(db.Float, default=5.0)


class LearningTopic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(80), nullable=False)
    sequence = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=True)
    youtube_url = db.Column(db.String(255), nullable=False)
    quiz_data_json = db.Column(db.Text, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ==========================================
# 3. HELPER FUNCTIONS & INIT
# ==========================================

def init_db():
    db.create_all()

    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@university.edu', role='admin', subject='Administration')
        admin.set_password('admin123')
        db.session.add(admin)

    subject_faculties = [
        {'username': 'faculty_ds', 'email': 'ds_faculty@university.edu', 'subject': 'Data Structures & Algorithms'},
        {'username': 'faculty_dbms', 'email': 'dbms_faculty@university.edu', 'subject': 'Database Management Systems'},
        {'username': 'faculty_os', 'email': 'os_faculty@university.edu', 'subject': 'Operating Systems'},
        {'username': 'faculty_ai', 'email': 'ai_faculty@university.edu', 'subject': 'Artificial Intelligence'}
    ]

    for f_data in subject_faculties:
        if not User.query.filter_by(username=f_data['username']).first():
            f_user = User(username=f_data['username'], email=f_data['email'], role='faculty', subject=f_data['subject'])
            f_user.set_password('faculty123')
            db.session.add(f_user)

    student = User.query.filter_by(username='student1').first()
    if not student:
        student = User(username='student1', email='student1@university.edu', role='student')
        student.set_password('student123')
        db.session.add(student)
        db.session.flush()

        grade_ds = StudentSubjectGrade(student_id=student.id, subject='Data Structures & Algorithms', attendance=85.0, midterm=78.0, assignment=80.0, logins=15.0, study_hours=6.0)
        grade_dbms = StudentSubjectGrade(student_id=student.id, subject='Database Management Systems', attendance=48.0, midterm=38.0, assignment=42.0, logins=6.0, study_hours=2.0)
        db.session.add_all([grade_ds, grade_dbms])

    db.session.commit()

with app.app_context():
    init_db()


def build_unique_fallback_quiz(topic_title, subject):
    topic_questions = [
        {"q": f"What primary problem is addressed in {topic_title}?", "a": "A) Data Organization & Efficiency", "opts": ["A) Data Organization & Efficiency", "B) UI Redesign", "C) Network Bandwidth", "D) Operating System Kernel"]},
        {"q": f"Which time complexity is ideal for operations in {topic_title}?", "a": "A) O(1) Constant Time", "opts": ["A) O(1) Constant Time", "B) O(n^2) Quadratic", "C) O(n!) Factorial", "D) O(2^n) Exponential"]},
        {"q": f"How is memory allocated for topics covered in {topic_title}?", "a": "A) Dynamic Heap / Contiguous Memory", "opts": ["A) Dynamic Heap / Contiguous Memory", "B) Static Read-Only Cache", "C) Virtual ROM Allocation", "D) GPU Framebuffers"]},
        {"q": f"What is a main trade-off when optimizing {topic_title}?", "a": "A) Time vs. Space Complexity", "opts": ["A) Time vs. Space Complexity", "B) Screen Resolution", "C) Network Latency", "D) Keyboard Input Delay"]},
        {"q": f"Which fundamental data structure supports operations in {topic_title}?", "a": "A) Arrays & Pointers", "opts": ["A) Arrays & Pointers", "B) CSS Selectors", "C) DOM Nodes", "D) Machine Code Bytes"]},
        {"q": f"What is the worst-case time complexity of searching in an unindexed structure in {topic_title}?", "a": "A) O(N) Linear Search", "opts": ["A) O(N) Linear Search", "B) O(1)", "C) O(log N)", "D) O(N log N)"]},
        {"q": f"How are duplicate entries handled logically in {topic_title}?", "a": "A) Explicit Key Checks or Hashing", "opts": ["A) Explicit Key Checks or Hashing", "B) Automatic OS Garbage Collection", "C) Stack Overflow Exceptions", "D) CPU Thread Halting"]},
        {"q": f"Which traversal or access method is most effective for {topic_title}?", "a": "A) Sequential or Indexed Access", "opts": ["A) Sequential or Indexed Access", "B) Random Bus Interrupts", "C) DMA Channel Allocation", "D) BIOS Boot Protocols"]},
        {"q": f"What happens during an overflow error in {topic_title}?", "a": "A) Allocated Capacity Exceeded", "opts": ["A) Allocated Capacity Exceeded", "B) Monitor Refresh Rate Drops", "C) User Session Logs Out", "D) Database Connection Closes"]},
        {"q": f"Why is {topic_title} essential in software development?", "a": "A) Optimizes Resource Utilization & Performance", "opts": ["A) Optimizes Resource Utilization & Performance", "B) Formats HTML Elements", "C) Compresses JPG Images", "D) Manages Wi-Fi Signals"]}
    ]
    
    formatted_quiz = []
    for i, item in enumerate(topic_questions):
        formatted_quiz.append({
            "id": i + 1,
            "question": f"Q{i+1}: {item['q']}",
            "options": item['opts'],
            "correct_answer": item['a'],
            "explanation": f"Core technical principle in {subject} - {topic_title}."
        })
    return formatted_quiz


# ==========================================
# 4. PORTAL & REPORT ROUTES
# ==========================================

@app.route('/')
def home():
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


@app.route('/upload_batch', methods=['POST'])
@login_required
def upload_batch():
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

        X = df[required_cols]
        X_scaled = scaler.transform(X) if scaler else X
        predictions = model.predict(X_scaled) if model else [0] * len(df)
        probabilities = model.predict_proba(X_scaled)[:, 1] if model and hasattr(model, 'predict_proba') else predictions

        faculty_subject = getattr(current_user, 'subject', 'General')
        existing_users = {u.username: u for u in User.query.all()}
        default_password_hash = generate_password_hash('student123')

        batch_results = []

        for idx, row in df.iterrows():
            risk_val = round(float(probabilities[idx]) * 100, 1)
            raw_name = str(row.get('name', f'student_{idx+1}')).strip()
            student_username = raw_name.replace(' ', '_').lower()
            student_email = str(row.get('email', f'{student_username}@university.edu'))
            student_id = str(row.get('id', f'STU-{idx+1000}'))

            if student_username in existing_users:
                student_user = existing_users[student_username]
            else:
                student_user = User(
                    username=student_username,
                    email=student_email,
                    role='student',
                    password_hash=default_password_hash
                )
                db.session.add(student_user)
                db.session.flush()
                existing_users[student_username] = student_user

            grade_entry = StudentSubjectGrade.query.filter_by(
                student_id=student_user.id, subject=faculty_subject
            ).first()

            if not grade_entry:
                grade_entry = StudentSubjectGrade(
                    student_id=student_user.id,
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
                'risk_percent': risk_val,
                'at_risk': bool(predictions[idx] == 1 or risk_val >= 50)
            })

        db.session.commit()
        last_batch_results = batch_results

        flash(f'Processed batch CSV for [{faculty_subject}]. Accounts and grades saved.', 'success')

        return render_template(
            'index.html',
            user=current_user,
            batch_results=batch_results,
            total_students=len(batch_results),
            at_risk_count=sum(1 for s in batch_results if s['at_risk'])
        )

    except Exception as e:
        return render_template('index.html', user=current_user, batch_error=f'Error processing CSV: {str(e)}')


# ==========================================
# 5. DOWNLOAD PDF & SEND ALERTS FEATURES
# ==========================================

@app.route('/download_report')
@login_required
def download_report():
    """Generates a PDF report of class risk analysis using ReportLab."""
    global last_batch_results
    if not last_batch_results:
        flash("No batch data available to generate report. Please process a CSV first.", "error")
        return redirect(url_for('home'))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#4F46E5'), spaceAfter=12
    )
    story.append(Paragraph("EduPredict AI — Class Risk Analysis Advisory", title_style))
    story.append(Paragraph(f"Faculty: {current_user.username} | Subject: {getattr(current_user, 'subject', 'General')}", styles['Normal']))
    story.append(Spacer(1, 12))

    data = [["Student ID", "Name", "Attendance", "Midterm", "Assignment", "Risk %", "Status"]]
    for s in last_batch_results:
        status_text = "AT RISK" if s['at_risk'] else "ON TRACK"
        data.append([
            s['id'], s['name'], f"{s['attendance']}%", f"{s['midterm']}%",
            f"{s['assignment']}%", f"{s['risk_percent']}%", status_text
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9FAFB')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
    ]))
    
    story.append(table)
    doc.build(story)
    
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Class_Risk_Report.pdf", mimetype="application/pdf")


@app.route('/send_alerts')
@login_required
def send_alerts():
    """Sends email alerts to students flagged as At Risk."""
    global last_batch_results
    if not last_batch_results:
        flash("No student batch results available to send alerts.", "error")
        return redirect(url_for('home'))

    at_risk_students = [s for s in last_batch_results if s['at_risk']]
    sent_count = 0

    for student in at_risk_students:
        try:
            msg = Message(
                subject="⚠️ Academic Performance Warning & Support Notice",
                recipients=[student['email']],
                body=f"""Hello {student['name']},

This is an automated academic advisory alert from EduPredict AI.

Your current aggregate failure risk for [{student['subject']}] is {student['risk_percent']}%.

Please log in to your Student Portal (http://127.0.0.1:5000/student_dashboard) to review your lecture video modules, take your 10-question practice quizzes, and connect with faculty assistance.

Best regards,
EduPredict AI Advisory Team
"""
            )
            mail.send(msg)
            sent_count += 1
        except Exception as err:
            print(f"Mail send error for {student['email']}: {err}")

    flash(f"Dispatched email alerts to {sent_count} flagged student(s)!", "success")
    return redirect(url_for('home'))


@app.route('/upload_teacher_video', methods=['POST'])
@login_required
def upload_teacher_video():
    if current_user.role not in ['faculty', 'admin']:
        return "Unauthorized Access", 403

    try:
        file = request.files.get('video_file')
        subject = request.form.get('subject', getattr(current_user, 'subject', 'General'))
        title = request.form.get('topic_title', 'Faculty Lecture')
        description = request.form.get('description', 'Teacher uploaded video.')

        if not file or file.filename == '':
            flash('Please select a valid video file (.mp4, .webm).', 'error')
            return redirect(url_for('home'))

        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(local_path)
        video_url = f"/static/uploads/videos/{filename}"

        questions = []
        if ai_client:
            try:
                gemini_file = ai_client.files.upload(file=local_path)
                while not gemini_file.state or gemini_file.state.name != "ACTIVE":
                    time.sleep(3)
                    gemini_file = ai_client.files.get(name=gemini_file.name)

                prompt = f"Scan lecture video '{title}' for '{subject}'. Generate 10 distinct multiple choice questions. Return raw JSON array of objects with id, question, options, correct_answer, explanation."
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[gemini_file, prompt]
                )

                if response and response.text:
                    clean_text = response.text.replace("```json", "").replace("```", "").strip()
                    questions = json.loads(clean_text)
            except Exception as e:
                print(f"Gemini API Error: {e}")

        if not questions or len(questions) < 10:
            questions = build_unique_fallback_quiz(title, subject)

        seq_count = LearningTopic.query.filter_by(subject=subject).count() + 1
        new_topic = LearningTopic(
            subject=subject,
            sequence=seq_count,
            title=f"Module {seq_count}: {title}",
            description=description,
            youtube_url=video_url,
            quiz_data_json=json.dumps(questions)
        )
        db.session.add(new_topic)
        db.session.commit()

        flash(f'Successfully added Module {seq_count} video to [{subject}] playlist! 10 distinct questions generated.', 'success')
        return redirect(url_for('home'))

    except Exception as e:
        flash(f'Error uploading video: {str(e)}', 'error')
        return redirect(url_for('home'))


@app.route('/student_dashboard')
@login_required
def student_dashboard():
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

        total_risk += risk_val

        topics = LearningTopic.query.filter_by(subject=g.subject).order_by(LearningTopic.sequence.asc()).all()
        pathway_topics = [
            {
                'id': t.id,
                'sequence': t.sequence,
                'title': t.title,
                'description': t.description,
                'youtube_url': t.youtube_url
            } for t in topics
        ]

        subject_cards.append({
            'subject': g.subject,
            'attendance': g.attendance,
            'midterm': g.midterm,
            'assignment': g.assignment,
            'study_hours': g.study_hours,
            'risk_percent': risk_val,
            'at_risk': at_risk,
            'pathway_topics': pathway_topics
        })

    overall_risk = round(total_risk / len(subject_cards), 2) if subject_cards else 25.0
    overall_at_risk = bool(overall_risk >= 50)

    return render_template('student.html', student=current_user, subject_cards=subject_cards, risk_percent=overall_risk, at_risk=overall_at_risk)


@app.route('/generate_topic_quiz', methods=['POST'])
@login_required
def generate_topic_quiz():
    try:
        data = request.json or {}
        topic_title = data.get('topic_title', '')
        subject = data.get('subject', '')

        topic = LearningTopic.query.filter_by(subject=subject, title=topic_title).first()
        if topic and topic.quiz_data_json:
            questions = json.loads(topic.quiz_data_json)
            return jsonify({'success': True, 'questions': questions})

        questions = build_unique_fallback_quiz(topic_title, subject)
        return jsonify({'success': True, 'questions': questions})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)