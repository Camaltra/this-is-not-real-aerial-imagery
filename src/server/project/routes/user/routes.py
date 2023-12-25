from project.models.user import User
from flask import g
from project.routes.user import user_bp
from flask import session as redis_session
from flask import abort, jsonify, request
from project.routes.user.utils import check_user_permission
import logging
from project.routes.user.schemas import GetUserParams, UpdateUserRequest
from pydantic import ValidationError


logger = logging.getLogger("UserEndPoint")
logger.setLevel(logging.WARNING)


@user_bp.route("/healthy")
def user_healthy():
    return "User"


@user_bp.route("/me")
def get_me():
    user_id = redis_session.get("user_id")

    user = g.session.query(User).filter(User.id == user_id).first()

    if user is None:
        abort(404, description="User not found")

    return jsonify(
        {
            "success": True,
            "object": {
                "type": "User",
                "id": user.id,
                "username": user.username,
                "email": user.email,
            },
        }
    )


@user_bp.route("/<user_id>")
def get_user_id(user_id: str):
    try:
        request_data = GetUserParams(user_id=user_id)
    except ValidationError as e:
        return jsonify({"error": "Invalid input data", "details": str(e.errors())}), 400

    user_id = request_data.user_id

    user = g.session.query(User).filter(User.id == user_id).first()
    if user is None:
        abort(404, description="User not found")
    return jsonify(
        {
            "success": True,
            "object": {
                "type": "User",
                "id": user.id,
                "username": user.username,
                # To see if email is really needed in the response
                "email": user.email,
            },
        }
    )


@user_bp.route("/")
def get_all_users():
    users = g.session.query(User).all()

    users_list = [
        {
            "id": user.id,
            "username": user.username,
        }
        for user in users
    ]

    return jsonify(success=True, users=users_list)


@user_bp.route("/<user_id>", methods=["PATCH"])
def update_user(user_id: int):
    session_user_id = redis_session.get("user_id")

    if not check_user_permission(session_user_id, user_id):
        abort(401, "Unauthorized")

    user_to_patch = g.session.query(User).filter(User.id == user_id).first()

    if not user_to_patch:
        abort(404, "User not found")

    request_data = UpdateUserRequest(**request.json)

    if request_data.username and request_data.username != user_to_patch.username:
        user_to_patch.username = request_data.username
    if request_data.email and request_data.email != user_to_patch.email:
        user_to_patch.email = request_data.email

    try:
        g.session.commit()
    except Exception as e:
        g.session.rollback()
        logger.error(f"Error updating user (ID: {user_id}): {str(e)}")
        abort(500, "Error updating user")

    return jsonify(
        {
            "success": True,
            "object": {
                "type": "User",
                "id": user_to_patch.id,
                "username": user_to_patch.username,
                "email": user_to_patch.email,
            },
        }
    )


@user_bp.route("/<user_id>", methods=["DELETE"])
def delete_user(user_id: str):
    session_user_id = redis_session.get("user_id")

    if not check_user_permission(session_user_id, user_id):
        abort(401, "Unauthorized")

    user_to_delete = g.session.query(User).filter(User.id == user_id).first()

    if user_to_delete.admin:
        abort(403, "Cannot delete admin user")

    if not user_to_delete:
        abort(404, "User not found")

    try:
        g.session.delete(user_to_delete)
        g.session.commit()
        return jsonify({"success": True, "message": "User deleted"})
    except Exception as e:
        g.session.rollback()
        logger.error(f"Error deleting user (ID: {user_id}): {str(e)}")
        abort(500, "Error deleting user")
