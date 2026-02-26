from flask import (
        Blueprint,render_template, request, url_for,redirect,flash, session, g, abort
    )

from werkzeug.security import generate_password_hash, check_password_hash

import functools
from functools import wraps

from .models import User
from app import db
from sqlalchemy import or_, func

bp = Blueprint ('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username'].lower()
        email = request.form['email'].lower()
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        institution = request.form['institution']

        error = None

        if User.query.filter_by(username=username).first():
            error = f'User {username} already exists'
        elif User.query.filter_by(email=email).first():
            error = f'Email {email} is already registered'

        if error is None:
            # FIRST ADMIN USER LOGIC:
            # Count how many users exist in the database
            # If it is the first one (count == 0), assign is_admin = True

            user_count = User.query.count()
            is_admin_role = True if user_count == 0 else False

            user = User(
                username=username,
                email=email,
                password=generate_password_hash(password),
                first_name=first_name,
                last_name=last_name,
                institution=institution,
                is_admin=is_admin_role
            )

            db.session.add(user)
            db.session.commit()
            return redirect(url_for('auth.login'))

        flash(error)

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        login_input = request.form['username'].lower()
        password = request.form['password']

        error = None

        user = User.query.filter(
            or_(
                func.lower(User.username) == login_input,
                func.lower(User.email) == login_input
            )
        ).first()

        if user is None or check_password_hash(user.password, password) is False:
            error = 'Incorrect username or password'
        elif user.is_enabled is False:
            error = 'This account is disabled. Please contact an administrator.'

        if error is None:
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('view.list'))

        flash(error)

    return render_template('auth/login.html')

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get_or_404(user_id)

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # If there is no user or the user is not an admin, we return a 403 (Forbidden) error.
        if g.user is None or not g.user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function