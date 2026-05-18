from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    admin = User.query.filter_by(is_admin=True).first()
    if not admin:
        new_admin = User(
            username='admin',
            email='admin@forensic.com',
            password=generate_password_hash('admin123'),
            is_admin=True
        )
        db.session.add(new_admin)
        db.session.commit()
        print("Admin user created: admin / admin123")
    else:
        print(f"Admin already exists: {admin.username}")
