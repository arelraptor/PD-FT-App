from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.Text, nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User {self.username}>'


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_by = db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    state = db.Column(db.Integer, default=0, nullable=False)
    prediction = db.Column(db.Integer)

    def __init__(self, created_by, title, description, prediction, state = 0):
        self.created_by = created_by
        self.title = title
        self.description = description
        self.state = state
        self.prediction = prediction

    def __repr__(self):
        return f'<Video {self.title}>'