from project.models.user import User
from flask import g
from project.routes.user import user_bp


@user_bp.route("/")
def user_healthy():
    return "User"

# @user_bp.route("/me")
# def users():
#     user = g.session.query(User).first()
#     if user is None:
#         return "No user"
#     return user.email
