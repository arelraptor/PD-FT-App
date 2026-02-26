from flask import Blueprint, render_template, request, redirect, url_for, g, flash, send_from_directory

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
        file = request.files['videofile']
        if file and allowed_file(file.filename):
            # 1. Generar nombre único para el archivo FINAL (.mp4)
            base_original, ext = os.path.splitext(file.filename)
            target_relative_path = os.path.join('uploads', base_original + ".mp4")
            final_mp4_path = get_unique_name(target_relative_path)

            # 2. Guardar el archivo subido con un nombre temporal
            temp_path = final_mp4_path + ext
            file.save(temp_path)

            # 3. Conversión Síncrona (Esperamos a que FFmpeg termine)
            if ext.lower() != '.mp4':
                try:
                    # El flag '-y' sobreescribe, '-i' es entrada
                    # IMPORTANTE: subprocess.run bloquea hasta que termina la conversión
                    subprocess.run(['ffmpeg', '-y', '-i', temp_path, final_mp4_path],
                                   check=True, capture_output=True)
                    os.remove(temp_path)
                except Exception as e:
                    if os.path.exists(temp_path): os.remove(temp_path)
                    flash(f"Error en conversión FFmpeg: {e}")
                    return redirect(request.url)
            else:
                os.rename(temp_path, final_mp4_path)

            # 4. Datos para la DB
            description = request.form.get('description', '')
            title = os.path.basename(final_mp4_path)

            video = Video(g.user.id, title, description)
            db.session.add(video)
            db.session.commit()

            # 5. Ejecutar procesamiento (IMPORTANTE: Pasar la ruta relativa correcta)
            # PD_Assessment.py espera el nombre del archivo.
            # Asegúrate de que PD_Assessment busque en la carpeta 'uploads/'
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
    # 'uploads' es la carpeta donde guardas los archivos según tu función upload()
    # Asegúrate de que la ruta sea relativa a la raíz del proyecto o absoluta
    upload_path = os.path.join(os.getcwd(), 'uploads')
    return send_from_directory(upload_path, filename)

@bp.route('/help')
def help():
    return render_template('help.html')