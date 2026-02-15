from flask import Blueprint, render_template, request, redirect, url_for, g, flash

from app.auth import login_required
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
        # Check if the file part is present in the request
        if 'videofile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['videofile']

        # Validate file existence and extension
        if file and allowed_file(file.filename):
            # 1. Define physical path to save the file
            relative_path = os.path.join('uploads', file.filename)
            filename = get_unique_name(relative_path)

            file.save(filename)

            description = request.form['description']
            title = os.path.basename(filename)

            video = Video(g.user.id, title, description)
            db.session.add(video)
            db.session.commit()

            # Execute processing script
            subprocess.Popen([sys.executable, 'myscript.py', filename, str(video.id)])

            return redirect(url_for('view.list'))
        else:
            # If the file is invalid, alert the user and block upload
            flash('Invalid file type. Only .mov, .mkv, .mp4, and .avi are allowed.')
            return redirect(request.url)

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