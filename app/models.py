from app import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Identification and Login
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.Text, nullable=False)

    # Personal Information
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    institution = db.Column(db.String(100), nullable=False)

    def __init__(self, username, email, password, first_name, last_name, institution):
        # Initialize the user instance with all required fields
        self.username = username
        self.email = email
        self.password = password
        self.first_name = first_name
        self.last_name = last_name
        self.institution = institution

    def __repr__(self):
        # Username remains the primary identifier for debugging purposes
        return f'<User {self.username}>'


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_by = db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    state = db.Column(db.Integer, default=0, nullable=False)
    prediction = db.Column(db.Integer)
    visible = db.Column(db.Boolean,default=True,nullable=False)

    def __init__(self, created_by, title, description, state = 0, visible = True):
        self.created_by = created_by
        self.title = title
        self.description = description
        self.state = state
        self.visible = visible

    def __repr__(self):
        return f'<Video {self.title}>'