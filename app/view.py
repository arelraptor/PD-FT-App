from flask import Blueprint, render_template, request, redirect, url_for, g, flash, send_from_directory
from werkzeug.security import generate_password_hash

from app.auth import login_required, admin_required
from .models import Video, User
from app import db

import os
import sys

import subprocess

# List of allowed video extensions
ALLOWED_EXTENSIONS = {'mov', 'mkv', 'mp4', 'avi'}

def allowed_file(filename):
    """
    Checks if the file has an extension and if it's within the allowed list.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video(id):
    video = Video.query.get_or_404(id)
    return video

def get_unique_name(uploaded_file):
    # Split name and extension
    base, ext = os.path.splitext(uploaded_file)
    counter = 1
    new_name = uploaded_file

    # While file exits, increase the counter
    while os.path.exists(new_name):
        new_name = f"{base}_{counter}{ext}"
        counter += 1

    return new_name

bp = Blueprint ('view', __name__, url_prefix='/view')

@bp.route('/list')
@login_required
def list():
    videos= Video.query.all()
    return render_template('view/list.html', videos=videos)


@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['videofile']
        if file and allowed_file(file.filename):
            base_original, ext = os.path.splitext(file.filename)
            target_relative_path = os.path.join('uploads', base_original + ".mp4")
            final_mp4_path = get_unique_name(target_relative_path)

            temp_path = final_mp4_path + ext
            file.save(temp_path)

            if ext.lower() != '.mp4':
                try:
                    subprocess.run(['ffmpeg', '-y', '-i', temp_path, final_mp4_path],
                                   check=True, capture_output=True)
                    os.remove(temp_path)
                except Exception as e:
                    if os.path.exists(temp_path): os.remove(temp_path)
                    flash(f"Error in FFmpeg conversion: {e}")
                    return redirect(request.url)
            else:
                os.rename(temp_path, final_mp4_path)

            description = request.form.get('description', '')
            title = os.path.basename(final_mp4_path)

            video = Video(g.user.id, title, description)
            db.session.add(video)
            db.session.commit()

            subprocess.Popen([sys.executable, 'PD_Assessment.py', final_mp4_path, str(video.id)])

            return redirect(url_for('view.list'))

    return render_template('view/upload.html')

@bp.route('/update/<int:id>', methods=['GET', 'POST'])
@login_required
def update(id):
    video = get_video(id)
    if request.method == 'POST':
        video.title = request.form['title']
        video.description = request.form['description']

        db.session.commit()

        return redirect(url_for('view.list'))
    return render_template('view/update.html',video=video)

@bp.route('/delete/<int:id>', methods=['GET', 'POST'])
@login_required
def delete(id):
    video = get_video(id)
    if request.method == 'POST':
        video.visible=0

        db.session.commit()

        return redirect(url_for('view.list'))
    return render_template('view/delete.html',video=video)

@bp.route('/play/<filename>')
@login_required
def play_video(filename):
    upload_path = os.path.join(os.getcwd(), 'uploads')
    return send_from_directory(upload_path, filename)

@bp.route('/help')
def help():
    return render_template('help.html')

@bp.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@bp.route('/admin/user/<int:id>/reset-password', methods=['POST'])
@admin_required
def reset_password(id):
    user = User.query.get_or_404(id)
    new_password = request.form.get('new_password')
    if new_password:
        user.password = generate_password_hash(new_password)
        db.session.commit()
        flash(f'Password updated for user {user.username}', 'success')
    return redirect(url_for('view.admin_users'))

@bp.route('/admin/user/<int:id>/toggle-admin', methods=['POST'])
@admin_required
def toggle_admin(id):
    user = User.query.get_or_404(id)
    if user.id == g.user.id:
        flash("You cannot remove your own admin status.")
    else:
        user.is_admin = not user.is_admin
        db.session.commit()
    return redirect(url_for('view.admin_users'))


@bp.route('/admin/user/create', methods=['POST'])
@admin_required
def admin_create_user():
    username = request.form['username'].lower()
    email = request.form['email'].lower()
    password = request.form['password']
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    institution = request.form['institution']
    is_admin = True if request.form.get('is_admin') else False

    error = None

    if User.query.filter_by(username=username).first():
        error = f'User {username} already exists'
    elif User.query.filter_by(email=email).first():
        error = f'Email {email} is already registered'

    if error is None:
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            institution=institution,
            is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()
        flash(f'User {username} created successfully.', 'success')
    else:
        flash(error)

    return redirect(url_for('view.admin_users'))


@bp.route('/admin/user/<int:id>/toggle-status', methods=['POST'])
@admin_required
def toggle_status(id):
    user = User.query.get_or_404(id)
    if user.id == g.user.id:
        flash("You cannot disable your own account.", "danger")
    else:
        user.is_enabled = not user.is_enabled
        db.session.commit()

        status_text = "enabled" if user.is_enabled else "disabled"
        flash(f"Status for {user.username} updated to {status_text}.", "success")

    return redirect(url_for('view.admin_users'))