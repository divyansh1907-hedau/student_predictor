from app import app, db, User

with app.app_context():
  db.create_all()

  # Create Default Admin / HOD
  if not User.query.filter_by(username='admin').first():
    admin = User(username='admin', email='hod@university.edu', role='admin')
    admin.set_password('admin123')
    db.session.add(admin)

  # Create Default Faculty
  if not User.query.filter_by(username='faculty').first():
    faculty = User(
        username='faculty', email='faculty@university.edu', role='faculty'
    )
    faculty.set_password('faculty123')
    db.session.add(faculty)

  # Create Default Demo Student
  if not User.query.filter_by(username='student1').first():
    student = User(
        username='student1',
        email='student1@university.edu',
        role='student',
        attendance=55.0,
        midterm=42.0,
        assignment=48.0,
        study_hours=3.5,
    )
    student.set_password('student123')
    db.session.add(student)

  db.session.commit()
  print('✅ Database Initialized Successfully with Demo Accounts!')