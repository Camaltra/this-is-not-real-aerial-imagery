from flask import Flask, g, jsonify
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from project.config import Config
from project.models.user import User
from project.routes.auth import auth_bp
from project.routes.user import user_bp
from project.extension import bcrypt
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException
from project.routes.quizz import quizz_bp


app = Flask(__name__)
app.config.from_object(Config)

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)

bcrypt.init_app(app)


@app.before_request
def before_request():
    g.session = Session()


@app.teardown_request
def teardown_request(exception=None):
    session = g.pop("session", None)
    if session is not None:
        session.close()


@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=str(e.description)), e.code

    elif isinstance(e, SQLAlchemyError):
        g.session.rollback()
        return jsonify(error="Database error"), 500

    else:
        return jsonify(error="Internal Server Error"), 500


app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(user_bp, url_prefix="/users")
app.register_blueprint(quizz_bp, url_prefix="/quizz")


@app.route("/")
def healthy():
    return "Healthy"


##############################################
### First Draft of Server Side Session Auth ##
##############################################

# @app.route("/@me")
# def get_current_user():
#     user_id = session.get("user_id")
#
#     if not user_id:
#         return jsonify({"error": "Unauthorized"}), 401
#
#     db_session = Session()
#
#     user = db_session.query(User).filter_by(id=user_id).first()
#     return jsonify({
#         "id": user.id,
#         "email": user.email
#     })
#
