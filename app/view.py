from flask import Blueprint, render_template, request, redirect, url_for, g

from app.auth import login_required
from .models import Video, User
from app import db

import os

import subprocess

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
            filename = os.path.join('uploads\\', file.filename)
            title=file.filename
            description = request.form['description']

            if os.path.exists(filename):
                dot_position_full_path=filename.rfind('.')
                filename=filename[0:dot_position_full_path]+'_2'+filename[dot_position_full_path:]
                dot_position_title = title.rfind('.')
                title = title[0:dot_position_title] + '_2' + title[dot_position_title:]

            file.save(filename)

            video = Video(g.user.id, title, description)
            db.session.add(video)
            db.session.commit()
            print("llamo a mi script")
            subprocess.Popen(['python', 'myscript.py',filename,str(video.id)])

            return redirect(url_for('view.list'))
    return render_template('view/upload.html')

def get_video(id):
    video = Video.query.get_or_404(id)
    return video

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