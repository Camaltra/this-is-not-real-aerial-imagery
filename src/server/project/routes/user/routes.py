from project.models.user import User
from flask import g
from project.routes.user import user_bp
from flask import session as redis_session
from flask import abort, jsonify, request
from project.routes.user.utils import check_user_permission


@user_bp.route("/healthy")
def user_healthy():
    return "User"


@user_bp.route("/me")
def get_me():
    user_id = redis_session.get("user_id")
    user = g.session.query(User).filter(User.id == user_id).first()
    if user is None:
        abort(404, description="User not found")
    return jsonify({"id": user.id, "username": user.username, "email": user.email})


@user_bp.route("/<user_id>")
def get_user_id(user_id: str):
    user = g.session.query(User).filter(User.id == user_id).first()
    if user is None:
        abort(404, description="User not found")
    return jsonify({"id": user.id, "username": user.username, "email": user.email})


@user_bp.route("/")
def get_all_users():
    users = g.session.query(User).all()
    if users is None or len(users) == 0:
        abort(404, description="Users not found")

    users_list = [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
        }
        for user in users
    ]
    return jsonify(users=users_list)


@user_bp.route("/<user_id>", methods=["PATCH"])
def update_user(user_id: int):
    session_user_id = redis_session.get("user_id")

    if not check_user_permission(session_user_id, user_id):
        abort(401, "Unauthorised")

    user_to_patch = g.session.query(User).filter(User.id == user_id).first()

    if (
        "username" in request.json
        and request.json.get("username") != user_to_patch.username
    ):
        user_to_patch.username = request.json.get("username")
    if "email" in request.json and request.json.get("email") != user_to_patch.email:
        user_to_patch.email = request.json.get("email")

    g.session.commit()

    return jsonify(
        {
            "id": user_to_patch.id,
            "username": user_to_patch.username,
            "email": user_to_patch.email,
        }
    )


@user_bp.route("/<user_id>", methods=["DELETE"])
def delete_user(user_id: str):
    session_user_id = redis_session.get("user_id")

    if not check_user_permission(session_user_id, user_id):
        abort(401, "Unauthorised")

    g.session.query(User).filter(User.id == user_id).delete()
    g.session.commit()

    return jsonify({"action": "delete succesful"})
