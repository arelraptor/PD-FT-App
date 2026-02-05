from flask import Blueprint, render_template, request, redirect, url_for, g

from app.auth import login_required
from .models import Video, User
from app import db

import os

import subprocess

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

        if file:
            filename = os.path.join('uploads', file.filename)
            filename=get_unique_name(filename)

            file.save(filename)

            description = request.form['description']
            title = filename.split('\\')[-1]

            video = Video(g.user.id, title, description)
            db.session.add(video)
            db.session.commit()
            print("llamo a mi script")
            subprocess.Popen(['python', 'myscript.py',filename,str(video.id)])

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