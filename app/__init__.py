from flask import Flask, render_template

def create_app():
    app = Flask(__name__)

    #Proyect configuretion
    app.config.from_mapping(
        DEBUG=True,
        SECRET_KEY='dev',
    )

    #Blueprint registration
    from . import view
    app.register_blueprint(view.bp)

    from . import auth
    app.register_blueprint(auth.bp)

    @app.route('/')
    def index():
        return render_template('index.html')
    return app



#python.exe -m flask run
#$env:FLASK_APP = "app/__init__.py"
#$env:FLASK_ENV = "development"
#$env:FLASK_DEBUG = 1
#In folder C:/Users/javier.amo/PycharmProjects/PDApp