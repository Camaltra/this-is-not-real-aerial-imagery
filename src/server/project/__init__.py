from flask import Flask, jsonify, g
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from project.config import Config
from project.models.user import User


app = Flask(__name__)
app.config.from_object(Config)

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)


@app.before_request
def before_request():
    g.session = Session()


@app.teardown_request
def teardown_request(exception=None):
    session = g.pop('session', None)
    if session is not None:
        session.close()


@app.route("/")
def hello_world():
    user = g.session.query(User).first()
    return user.email


#
#
# from flask import Flask, request, jsonify, session
# from flask_bcrypt import Bcrypt
# from flask_cors import CORS
# from config import ApplicationConfig
# from server.data import Session
# from server.data import User
#
# app = Flask(__name__)
# app.config.from_object(ApplicationConfig)
# app.secret_key = ApplicationConfig.SECRET_KEY
#
# bcrypt = Bcrypt(app)
# CORS(app, supports_credentials=True)
#
#
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
#
# @app.route("/register", methods=["POST"])
# def register_user():
#     email = request.json["email"]
#     password = request.json["password"]
#
#     db_session = Session()
#
#     user_exists = db_session.query(User).filter_by(email=email).first() is not None
#
#     if user_exists:
#         return jsonify({"error": "User already exists"}), 409
#
#     hashed_password = bcrypt.generate_password_hash(password)
#     new_user = User(email=email, password=hashed_password)
#     db_session.add(new_user)
#     db_session.commit()
#
#     session["user_id"] = new_user.id
#
#     return jsonify({
#         "id": new_user.id,
#         "email": new_user.email
#     })
#
#
# @app.route("/login", methods=["POST"])
# def login_user():
#     email = request.json["email"]
#     password = request.json["password"]
#
#     db_session = Session()
#
#     user = db_session.query(User).filter_by(email=email).first()
#
#     if user is None:
#         return jsonify({"error": "Unauthorized"}), 401
#
#     if not bcrypt.check_password_hash(user.password, password):
#         return jsonify({"error": "Unauthorized"}), 401
#
#     session["user_id"] = user.id
#
#     return jsonify({
#         "id": user.id,
#         "email": user.email
#     })
#
#
# @app.route("/logout", methods=["POST"])
# def logout_user():
#     session.pop("user_id")
#     return "200"
#
#
# @app.route("/delete_quizz", methods=["POST"])
# def create_quizz():
#     # request.json("date") or request.json("id")
#     # Delete a quizz
#     # Move pics from quizz folder to own folder
#     # Delete quizz
#     return True
#
#
# @app.route("/delete_quizz", methods=["POST"])
# def delete_quizz():
#     # Delete a quizz
#     return True
#
# @app.route("/disable_quizz", methods=["POST"])
# def disable_quizz():
#     # Disable a quizz
#     return True
#
# # Commit
# # Return Quizz ID
#
#
# @app.route("/")
# def home():
#     return ApplicationConfig.SQLALCHEMY_DATABASE_URI
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
