import io

from flask import g, session
from project.routes.quizz import quizz_bp
from flask import abort, jsonify, request, send_file
from sqlalchemy import or_
from project.models import PictureQuestion, Quizz, UserQuizzHistory
from project.routes.user.utils import is_admin
from project.connectors.s3 import S3Manager
from project.routes.quizz.constant import BUCKET_NAME
import time
from project.routes.quizz.utils import QuizzFileManager, build_quizz_content
import logging

logger = logging.getLogger("QuizzEndPoint")
logger.setLevel(logging.WARNING)


@quizz_bp.route("/healthy")
def quizz_healthy():
    return "Quizz"


@quizz_bp.route("/")
def get_available_quizz_title():
    available_quizzs = g.session.query(Quizz).filter(Quizz.available == True).all()
    quizzs = [{"id": quizz.id, "name": quizz.quizz_name} for quizz in available_quizzs]
    return jsonify(quizzs=quizzs)


@quizz_bp.route("/<quizz_id>/<question_ix>")
def get_quizz_picture(quizz_id: int, question_ix: int):
    quizz: Quizz | None = (
        g.session.query(Quizz)
        .filter(Quizz.available == True, Quizz.id == quizz_id)
        .first()
    )
    if quizz is None:
        abort(404, "quizz not found")

    if question_ix >= len(quizz.picture_questions):
        abort(404, "question not found")

    manager = S3Manager(BUCKET_NAME)
    image = quizz.picture_questions[question_ix]
    image_data = manager.get_object(image.filename)
    return send_file(
        io.BytesIO(image_data),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{quizz.id}_{image.id}",
    )


@quizz_bp.route("/", methods=["POST"])
def create_quizz():
    if not is_admin(session.get("user_id")):
        abort(401, "Unauthorized")

    manager = QuizzFileManager()

    number_of_pics = request.json.get("number_of_pics", 5)

    images, targets = build_quizz_content(manager, number_of_pics)

    quizz_name = request.json.get("quizz_name")
    if quizz_name is None:
        abort(400, "Missing quizz_name parameter")

    folder_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_{quizz_name}"

    created_quizz = Quizz(folder_name=folder_name, quizz_name=quizz_name)
    for image, target in zip(images, targets):
        created_quizz.picture_questions.append(
            PictureQuestion(
                filename=f"{created_quizz.folder_name}/{image.get('Key').split('/')[-1]}",
                old_filename=image.get("Key"),
                awnser=target,
            )
        )

    try:
        g.session.add(created_quizz)
        g.session.commit()
    except Exception as e:
        g.session.rollback()
        logger.warning(f"SQLAlchemyException: {e}")
        return jsonify({"error": "Error while created the quizz"})

    manager.build_quizz_folder(created_quizz)
    return jsonify(
        {
            "success": "created",
            "object": {
                "type": "Quizz",
                "quizz_id": created_quizz.id,
                "quizz_name": created_quizz.quizz_name,
            },
        }
    )


@quizz_bp.route("/compute_score", methods=["POST"])
def compute_score():
    user_awnser = request.json.get("user_awnser")
    quizz_id = request.json.get("quizz_id")
    question_ix = request.json.get("question_ix")
    if user_awnser is None or quizz_id is None or question_ix is None:
        abort(400, "missing parameter, please send user_awnser, quizz_id, question_ix")

    quizz: Quizz | None = g.session.query(Quizz).filter(Quizz.id == quizz_id).first()
    if quizz is None:
        abort(404, "quizz not found")

    if question_ix >= len(quizz.picture_questions):
        abort(404, "picture not found")

    question = quizz.picture_questions[question_ix]
    return jsonify({"response": user_awnser == question.awnser})


@quizz_bp.route("/score_save", methods=["POST"])
def compute_scores_and_save():
    user_awnsers = request.json.get("user_awnsers")
    quizz_id = request.json.get("quizz_id")
    user_id = session.get("user_id")

    if user_awnsers is None or quizz_id is None:
        abort(400, "missing parameter, please send user_awnsers and quizz_id")

    quizz: Quizz | None = g.session.query(Quizz).filter(Quizz.id == quizz_id).first()
    if quizz is None:
        abort(404, "quizz not found")

    if len(user_awnsers) != len(quizz.picture_questions):
        abort(404, "awnser len doesn't match quizz len")
    score = 0
    for ix, user_awnser in enumerate(user_awnsers):
        if user_awnser == quizz.picture_questions[ix].awnser:
            score += 1
    score /= len(user_awnsers)

    score = UserQuizzHistory(user_id=user_id, quizz_id=quizz_id, score=score)
    try:
        g.session.add(score)
        g.session.commit()
    except Exception as e:
        g.session.rollback()
        logger.warning(f"SQLAlchemyException: {e}")
        return jsonify({"error": "Error while trying to save the score"})

    return jsonify({"success": "score computed and saved", "score": score})


@quizz_bp.route("/", methods=["DELETE"])
def delete_quizz():
    manager = QuizzFileManager()
    quizz_name = request.json.get("quizz_name")
    quizz_id = request.json.get("quizz_id")
    if quizz_name is None and quizz_id is None:
        abort(400, "Please give one of the following parameter: quizz_name | quizz_id")
    quizz_to_delete: Quizz | None = (
        g.session.query(Quizz)
        .filter(or_(Quizz.quizz_name == quizz_name, Quizz.id == quizz_id))
        .first()
    )
    if quizz_to_delete:
        try:
            g.session.delete(quizz_to_delete)
            g.session.commit()
        except Exception as e:
            g.session.rollback()
            logger.warning(f"SQLAlchemyException: {e}")
            return jsonify({"error": "error while deleting the folder"})
        manager.delete_quizz_folder(quizz_to_delete)
        return jsonify(
            {
                "success": "deleted",
                "object": {
                    "type": "Quizz",
                    "quizz_id": quizz_to_delete.id,
                    "quizz_name": quizz_to_delete.quizz_name,
                },
            }
        )
    return jsonify({"no operation": "quizz not found"})
