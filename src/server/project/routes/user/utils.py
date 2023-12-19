from project.models.user import User
from flask import g


def is_admin(session_user_id: int) -> bool:
    user = g.session.query(User).filter(User.id == session_user_id).first()
    if user is None or not user.admin:
        return False
    return True


def check_user_permission(session_user_id: int, user_id: int) -> bool:
    """
    Check if user has permission, it checks:
    - if user resquest information, or modification on itself
    - if user is administrator
    :param session_user_id: The current connected User
    :param user_id: The requested user_id
    :return:
    """
    if session_user_id is None:
        return False

    if int(user_id) != session_user_id:
        user = g.session.query(User).filter(User.id == session_user_id).first()
        if user is None or not user.admin:
            return False

    return True
