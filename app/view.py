from flask import Blueprint

bp = Blueprint ('view', __name__, url_prefix='/view')

@bp.route('/list')
def index():
    return "Lista de tareas"

@bp.route('/upload')
def upload():
    return "Subir un video"