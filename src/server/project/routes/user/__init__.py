from flask import Blueprint

user_bp = Blueprint('user', __name__)

from project.routes.user import routes
