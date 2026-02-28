from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    uploads = db.relationship('UploadHistory', backref='owner', lazy=True)

class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False) # 'image' or 'video'
    result = db.Column(db.String(100), nullable=False)   # Detailed Category
    confidence_score = db.Column(db.Float, nullable=False)
    
    # New Fields for Advanced Audit
    real_prob = db.Column(db.Float, default=0.0)
    edited_prob = db.Column(db.Float, default=0.0)
    ai_prob = db.Column(db.Float, default=0.0)
    tool_inference = db.Column(db.String(255), default="None Detected")
    
    # Detailed Data (Stored as JSON string)
    # Includes metadata, region details, frame-by-frame scores for videos
    detailed_report = db.Column(db.Text, nullable=True)
    date_time = db.Column(db.DateTime, default=datetime.utcnow)

    def set_report(self, data):
        self.detailed_report = json.dumps(data)

    def get_report(self):
        return json.loads(self.detailed_report) if self.detailed_report else {}
