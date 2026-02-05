from flask import Blueprint, render_template, request, redirect, url_for, g

from app.auth import login_required
from .models import Video, User
from app import db

import os
import sys

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
            # 1. Definimos la ruta física para guardar el archivo
            relative_path = os.path.join('uploads', file.filename)
            filename = get_unique_name(relative_path)  # Esto sigue siendo "uploads/nombre.avi"

            file.save(filename)

            description = request.form['description']

            # 2. CORRECCIÓN: Usamos os.path.basename para obtener SOLO el nombre,
            # sin importar si hay / o \
            title = os.path.basename(filename)

            video = Video(g.user.id, title, description)
            db.session.add(video)
            db.session.commit()

            print("llamo a mi script")
            # 3. Importante: Para el script pasamos 'filename' (ruta completa),
            # pero para la DB guardamos 'title' (solo nombre).
            subprocess.Popen([sys.executable, 'myscript.py', filename, str(video.id)])

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