from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    db.drop_all()
    db.create_all()
    
    # Create default Admin
    admin = User(
        username='admin', 
        email='admin@forensic.com', 
        password=generate_password_hash('admin123'),
        is_admin=True
    )
    db.session.add(admin)
    db.session.commit()
    
    print("Database reset successful. Default Admin: admin / admin123")
