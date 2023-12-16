from project.routes.auth import auth_bp
from flask import session as redis_session
from flask import jsonify, request, g
from project.models.user import User
from project.extension import bcrypt

@auth_bp.route("/")
def auth_healthy():
    return "Auth"


@auth_bp.route("/register", methods=["POST"])
def register_user():
    email = request.json.get("email")
    password = request.json.get("password")
    username = request.json.get("username")

    user_exists = g.session.query(User).filter_by(email=email).first() is not None

    if user_exists:
        return jsonify({"error": "User already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(username=username, email=email, password=hashed_password.decode('utf8'))
    g.session.add(new_user)
    g.session.commit()

    redis_session["user_id"] = new_user.id

    return jsonify({
        "id": new_user.id,
        "email": new_user.email
    })


@auth_bp.route("/login", methods=["POST"])
def login_user():
    email = request.json.get("email")
    password = request.json.get("password")

    user = g.session.query(User).filter_by(email=email).first()

    if user is None:
        return jsonify({"error": "Unauthorized"}), 401

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401

    redis_session["user_id"] = user.id

    return jsonify({
        "id": user.id,
        "email": user.email
    })


@auth_bp.route("/logout", methods=["POST"])
def logout_user():
    redis_session.pop("user_id")
    return "200"
