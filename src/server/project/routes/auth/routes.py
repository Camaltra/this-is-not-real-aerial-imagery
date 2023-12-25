from project.routes.auth import auth_bp
from flask import session as redis_session
from flask import jsonify, request, g
from project.models.user import User
from project.extension import bcrypt
from project.routes.auth.utils import compute_password
from project.routes.auth.schemas import RegistrationModel, LoginModel
import logging
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger("AuthEndPoint")
logger.setLevel(logging.WARNING)


@auth_bp.route("/")
def auth_healthy():
    return "Auth"


@auth_bp.route("/register", methods=["POST"])
def register_user():
    try:
        data = RegistrationModel(**request.json)
    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

    password = data.password
    email = data.email

    try:
        hashed_password = compute_password(password)
    except ValueError as e:
        logger.warning(f"ValueError: {e}")
        return jsonify({"error": "Error while hashing the password"}), 500

    new_user = User(email=email, password=hashed_password)

    try:
        g.session.add(new_user)
        g.session.commit()
        redis_session["user_id"] = new_user.id
    except IntegrityError as e:
        g.session.rollback()
        logger.warning(f"IntegrityError: {e}")
        return jsonify({"error": "User already exists"}), 409
    except Exception as e:
        g.session.rollback()
        logger.warning(f"Unexpected error: {e}")
        return jsonify({"error": "Error while creating the user"}), 500

    return jsonify(
        {
            "success": True,
            "object": {"type": "User", "user_id": new_user.id, "email": new_user.email},
        }
    )


@auth_bp.route("/login", methods=["POST"])
def login_user():
    try:
        data = LoginModel(**request.json)
    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

    email = data.email
    password = data.password

    try:
        user = g.session.query(User).filter_by(email=email).first()
        if user is None:
            return jsonify({"error": "Unauthorized"}), 401

        if not bcrypt.check_password_hash(user.password, password):
            return jsonify({"error": "Unauthorized"}), 401

        redis_session["user_id"] = user.id
        return jsonify(
            {
                "success": True,
                "object": {
                    "type": "User",
                    "user_id": user.id,
                },
            }
        )
    except Exception as e:
        logger.warning(f"Unexpected error during login: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@auth_bp.route("/logout", methods=["DELETE"])
def logout_user():
    redis_session.pop("user_id")
    return jsonify({"success": True})
