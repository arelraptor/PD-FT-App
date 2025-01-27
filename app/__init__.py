from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    #Proyect configuration
    app.config.from_mapping(
        DEBUG=True,
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI = "sqlite:///ftpredictor.db"
    )

    db.init_app(app)

    #Blueprint registration
    from . import view
    app.register_blueprint(view.bp)

    from . import auth
    app.register_blueprint(auth.bp)

    @app.route('/')
    def index():
        return render_template('index.html')

    with app.app_context():
        db.create_all()

    return app



#python.exe -m flask run
#$env:FLASK_APP = "app/__init__.py"
#$env:FLASK_ENV = "development"
#$env:FLASK_DEBUG = 1
#In folder C:/Users/javier.amo/PycharmProjects/PDApp