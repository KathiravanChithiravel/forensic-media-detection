import os
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from models import db, User, UploadHistory
import forensics
from functools import wraps
from flask_mail import Mail, Message

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fake_media_v3_premium_secret_123')

# Use absolute path for SQLite to avoid issues in production
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ELA_FOLDER'] = os.path.join('static', 'ela_results')

# Initialize DB
db.init_app(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ELA_FOLDER'], exist_ok=True)

# Mail Configuration (Placeholders for USER to fill)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)

# Force DB creation but skip if it errors during startup context
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    logger.error(f"Startup DB Error: {e}")

# --- Decorators ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access restrictive: Administrator credentials required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('This username is already registered in our network.', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('This email is already registered.', 'danger')
        else:
            try:
                new_user = User(
                    username=username, 
                    email=email, 
                    password=generate_password_hash(password)
                )
                db.session.add(new_user)
                db.session.commit()
                flash('Account initialized successfully! Please sign in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                logger.error(f"Registration error: {e}")
                flash('An error occurred during registration. Please try again.', 'danger')
                
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            logger.info(f"User {username} logged in.")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please verify your identity.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been securely logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Verification link has been dispatched to your registered email.', 'info')
            return redirect(url_for('reset_password'))
        else:
            flash('Identity not found in our registry.', 'danger')
    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        flash('Success: Credentials updated. You can now login.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html')

@app.route('/dashboard')
@login_required
def dashboard():
    history = UploadHistory.query.filter_by(user_id=current_user.id).order_by(UploadHistory.date_time.desc()).all()
    
    # Calculate stats here to avoid Jinja2 template search errors
    stats = {
        'total': len(history),
        'fake': len([h for h in history if any(word in h.result for word in ['Fake', 'Deepfake', 'AI'])]),
        'suspicious': len([h for h in history if 'Suspicious' in h.result])
    }
    stats['real'] = stats['total'] - stats['fake'] - stats['suspicious']
    
    return render_template('dashboard.html', history=history, stats=stats)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for audit.', 'warning')
            return redirect(request.url)
            
        if file:
            try:
                # Generate unique filename
                filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Identify media type
                ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                file_type = 'image' if ext in ['png', 'jpg', 'jpeg', 'webp'] else 'video'
                
                if file_type == 'image':
                    # Perform Forensic Pipeline
                    metadata = forensics.extract_exif(filepath)
                    ela_filename = f"ela_{filename}"
                    ela_path = os.path.join(app.config['ELA_FOLDER'], ela_filename)
                    ela_np = forensics.perform_ela(filepath, ela_path)
                    
                    noise_filename = f"noise_{filename}"
                    forensics.perform_noise_map(filepath, os.path.join(app.config['ELA_FOLDER'], noise_filename))
                    clone_filename = f"clone_{filename}"
                    forensics.perform_clone_map(filepath, os.path.join(app.config['ELA_FOLDER'], clone_filename))
                    
                    is_noise_susc, noise_msg, _ = forensics.detect_noise_inconsistency(filepath)
                    is_blur_susc, blur_msg, _ = forensics.detect_blur_manipulation(filepath)
                    is_color_susc, color_msg, color_val = forensics.detect_color_inconsistency(filepath)
                    is_clone_susc, clone_msg, _ = forensics.detect_clone_stamp(filepath)
                    ai_prob, ai_reason = forensics.detect_ai_generation_image(filepath)
                    
                    # New Advanced Tests
                    is_hist_susc, hist_msg, _ = forensics.analyze_histogram_irregularity(filepath)
                    is_ring_susc, ring_msg, _ = forensics.detect_edge_ringing(filepath)
                    
                    regions = forensics.analyze_regions(filepath, ela_np)
                    
                    # Tool Inference
                    intermediate_res = {
                        'is_blur_susc': is_blur_susc, 
                        'is_clone_susc': is_clone_susc, 
                        'is_noise_susc': is_noise_susc,
                        'is_color_susc': is_color_susc,
                        'metadata': metadata
                    }
                    tool_inf = forensics.detect_tool_inference(filepath, intermediate_res)
                    
                    # Sentiment and Results
                    score_sum = 0
                    reasons = []
                    if not metadata.get('has_metadata'): score_sum += 20
                    if is_noise_susc: score_sum += 25; reasons.append(noise_msg)
                    if is_blur_susc: score_sum += 20; reasons.append(blur_msg)
                    if is_clone_susc: score_sum += 35; reasons.append(clone_msg)
                    if is_hist_susc: score_sum += 15; reasons.append(hist_msg)
                    if is_ring_susc: score_sum += 15; reasons.append(ring_msg)
                    if len([r for r in regions if r['is_suspicious']]) > 3: score_sum += 35
                    
                    ai_final = float(ai_prob)
                    edited_final = float(min(float(score_sum), 100.0))
                    
                    # Probability Distribution
                    if ai_final > 60:
                        prob_ai = ai_final
                        prob_edited = float(min(edited_final, 30.0))
                        prob_real = float(max(0.0, 100.0 - prob_ai - prob_edited))
                        result = "AI Generated / Deepfake (FAKE)"
                    else:
                        prob_edited = edited_final
                        prob_ai = ai_final
                        prob_real = float(max(0.0, 100.0 - prob_edited - (ai_final / 2.0)))
                        
                        total_p = prob_real + prob_edited + prob_ai
                        if total_p > 0:
                            prob_real = (prob_real / total_p) * 100.0
                            prob_edited = (prob_edited / total_p) * 100.0
                            prob_ai = (prob_ai / total_p) * 100.0
                        
                        # Classification logic
                        if prob_edited > 65: result = "Digitally Manipulated (FAKE)"
                        elif prob_edited > 35: result = "Suspicious / Likely Edited"
                        else: result = "Real Photograph"

                    confidence = max(prob_real, prob_edited, prob_ai)
                    
                    report_data = {
                        'type': 'image', 'filename': filename, 'ela_filename': ela_filename,
                        'noise_filename': noise_filename, 'clone_filename': clone_filename,
                        'metadata': metadata, 'regions': regions, 'reasons': reasons,
                        'tool_inference': tool_inf, 'ai_prob': ai_prob, 'ai_reason': ai_reason,
                        'is_noise_susc': is_noise_susc, 'is_blur_susc': is_blur_susc,
                        'is_clone_susc': is_clone_susc, 'is_color_susc': is_color_susc,
                        'is_hist_susc': is_hist_susc, 'is_ring_susc': is_ring_susc,
                        'prob_real': prob_real, 'prob_edited': prob_edited, 'prob_ai': prob_ai
                    }
                else:
                    # Video Forensic Pipeline
                    v = forensics.analyze_video(filepath)
                    result = v['result']
                    confidence = v['confidence']
                    prob_real, prob_edited, prob_ai = v['prob_real'], v['prob_edited'], v['prob_ai']
                    reasons = [f"Detected {v['suspicious_frames']} suspicious frames with pixel distortion."]
                    if prob_ai > 60: reasons.append("Motion and texture audit indicates high GAN probability.")
                    
                    report_data = {
                        'type': 'video', 'filename': filename, 'total_frames': v['total_frames'],
                        'analyzed_frames': v['analyzed_frames'], 'suspicious_frames': v['suspicious_frames'],
                        'avg_anomaly_score': v['avg_anomaly_score'], 'frame_scores': v['frame_scores'],
                        'prob_real': prob_real, 'prob_edited': prob_edited, 'prob_ai': prob_ai,
                        'reasons': reasons
                    }
                
                # Persist Audit in Registry
                new_upload = UploadHistory(
                    user_id=current_user.id, file_name=filename, file_type=file_type,
                    result=result, confidence_score=float(confidence),
                    real_prob=float(prob_real), edited_prob=float(prob_edited), ai_prob=float(prob_ai),
                    tool_inference=report_data.get('tool_inference', 'None') if file_type == 'image' else 'N/A'
                )
                new_upload.set_report(report_data)
                db.session.add(new_upload)
                db.session.commit()
                
                report_data.update({'result': result, 'confidence': float(confidence)})
                return render_template('result.html', data=report_data)
                
            except Exception as e:
                logger.error(f"Upload processing failed: {e}")
                flash(f'Audit failed: {str(e)}', 'danger')
                return redirect(request.url)
                
    return render_template('upload.html')

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    audit = UploadHistory.query.get_or_404(report_id)
    # Security: Only owner or admin can view
    if not current_user.is_admin and audit.user_id != current_user.id:
        flash("Authorization Restricted: You do not have permissions to view this audit.", "danger")
        return redirect(url_for('dashboard'))
    
    data = audit.get_report()
    data.update({
        'id': audit.id,
        'result': audit.result,
        'confidence': audit.confidence_score,
        'archived': True
    })
    return render_template('result.html', data=data)

@app.route('/send_email/<int:report_id>')
@login_required
def send_report_email(report_id):
    audit = UploadHistory.query.get_or_404(report_id)
    if not current_user.is_admin and audit.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
    try:
        msg = Message(f"Forensic Audit Report: {audit.result}",
                      recipients=[current_user.email])
        msg.body = f"""
        AI Forensic Audit Report Summary
        -------------------------------
        File: {audit.file_name}
        Result: {audit.result}
        Confidence: {audit.confidence_score}%
        
        Detailed Matrix:
        - Real Probability: {audit.real_prob}%
        - Edited Probability: {audit.edited_prob}%
        - AI Probability: {audit.ai_prob}%
        
        Audit generated on: {audit.date_time}
        """
        mail.send(msg)
        return jsonify({'success': True, 'message': 'Report dispatched to your email!'})
    except Exception as e:
        import traceback
        logger.error(f"Email failure: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Email error: {str(e)}'}), 500

# --- Admin Routes ---

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    users = User.query.all()
    uploads = UploadHistory.query.order_by(UploadHistory.date_time.desc()).all()
    
    stats = {
        'total_users': len(users),
        'total_audits': len(uploads),
        'image_audits': UploadHistory.query.filter_by(file_type='image').count(),
        'video_audits': UploadHistory.query.filter_by(file_type='video').count(),
    }
    return render_template('admin_dashboard.html', users=users, uploads=uploads, stats=stats)

@app.route('/admin/delete_user/<int:id>')
@admin_required
def delete_user(id):
    user = User.query.get_or_404(id)
    if user.is_admin:
        flash("Authorization Restricted: Local governance cannot delete admin accounts.", "danger")
    else:
        UploadHistory.query.filter_by(user_id=id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"Investigator {user.username} data wipe completed.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_upload/<int:id>')
@admin_required
def delete_upload(id):
    audit = UploadHistory.query.get_or_404(id)
    db.session.delete(audit)
    db.session.commit()
    flash("Audit record removed from global registry.", "info")
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    # Bind to 0.0.0.0 to allow access from local network
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

