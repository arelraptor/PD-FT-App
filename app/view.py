from flask import Blueprint, render_template, request, redirect, url_for, g

from app.auth import login_required
from .models import Video, User
from app import db

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
        title = request.form['title']
        description = request.form['description']
        video = Video(g.user.id, title, description)

        db.session.add(video)
        db.session.commit()
        return redirect(url_for('view.list'))
    return render_template('view/upload.html')

def get_video(id):
    print(id)
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