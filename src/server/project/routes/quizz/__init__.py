from flask import Blueprint

quizz_bp = Blueprint("quizz", __name__)

from project.routes.quizz import routes
