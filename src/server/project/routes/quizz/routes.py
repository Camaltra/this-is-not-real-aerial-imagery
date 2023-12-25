import io

from flask import g, session
from project.routes.quizz import quizz_bp
from flask import abort, jsonify, request, send_file
from sqlalchemy import or_
from sqlalchemy.sql.expression import true
from project.models import PictureQuestion, Quizz, UserQuizzHistory
from project.routes.user.utils import is_admin
from project.connectors.s3 import S3Manager
from project.routes.quizz.constant import BUCKET_NAME
import time
from project.routes.quizz.utils import QuizzFileManager, build_quizz_content
import logging
from project.routes.quizz.schemas import (
    QuizzPictureRequest,
    QuizzCreateRequest,
    ComputeScoresAndSaveParams,
    DeleteQuizzParams,
)
from pydantic import ValidationError
from sqlalchemy import exc
from project.routes.quizz.exceptions import InsufficientImagesError


logger = logging.getLogger("QuizzEndPoint")
logger.setLevel(logging.WARNING)


@quizz_bp.route("/healthy")
def quizz_healthy():
    return "Quizz"


@quizz_bp.route("/")
def get_available_quizz_title():
    available_quizzs = g.session.query(Quizz).filter(Quizz.available == true()).all()
    quizzs = [{"id": quizz.id, "name": quizz.quizz_name} for quizz in available_quizzs]
    return jsonify(quizzs=quizzs)


@quizz_bp.route("/<quizz_id>/<question_ix>")
def get_quizz_picture(quizz_id: int, question_ix: int):
    try:
        request_data = QuizzPictureRequest(quizz_id=quizz_id, question_ix=question_ix)
    except ValidationError as e:
        return jsonify({"error": "Invalid input data", "details": str(e.errors())}), 400

    quizz_id = request_data.quizz_id
    question_ix = request_data.question_ix

    quizz: Quizz | None = (
        g.session.query(Quizz)
        .filter(Quizz.available == true(), Quizz.id == quizz_id)
        .first()
    )
    if quizz is None:
        abort(404, f"Quiz with id {quizz_id} not found")

    if question_ix >= len(quizz.picture_questions):
        abort(404, f"Question {question_ix} not found in quiz with id {quizz_id}")

    manager = S3Manager(BUCKET_NAME)
    image = quizz.picture_questions[question_ix]
    image_data = manager.get_object(image.filename)
    return send_file(
        io.BytesIO(image_data),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{quizz.id}_{image.id}",
    )


@quizz_bp.route("/create_quizz", methods=["POST"])
def create_quizz():
    if not is_admin(session.get("user_id")):
        abort(401, "Unauthorized")

    try:
        request_data = QuizzCreateRequest(**request.json)
    except ValidationError as e:
        return jsonify({"error": "Invalid input data", "details": str(e.errors())}), 400

    manager = QuizzFileManager()

    quizz_name = request_data.quizz_name
    number_of_pics = request_data.number_of_pics

    try:
        images, targets = build_quizz_content(manager, number_of_pics)
    except InsufficientImagesError as e:
        return jsonify({"error": str(e)}), 400

    if quizz_name is None:
        return jsonify({"error": "Missing quizz_name parameter"}), 400

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
    except exc.SQLAlchemyError as e:
        g.session.rollback()
        logger.error(f"Failed to create quiz. SQLAlchemyError: {e}")
        return jsonify({"error": "Error while creating the quiz"}), 500

    manager.build_quizz_folder(created_quizz)
    return jsonify(
        {
            "success": True,
            "object": {
                "type": "Quizz",
                "quizz_id": created_quizz.id,
                "quizz_name": created_quizz.quizz_name,
                "folder_name": created_quizz.folder_name,
            },
        }
    )


@quizz_bp.route("/score_save", methods=["POST"])
def compute_scores_and_save():
    try:
        request_data = ComputeScoresAndSaveParams(**request.json)
    except ValidationError as e:
        return jsonify({"error": "Invalid input data", "details": str(e.errors())}), 400

    user_id = request_data.user_id
    quizz_id = request_data.quizz_id
    user_answers = request_data.user_answers

    quizz: Quizz | None = g.session.query(Quizz).filter(Quizz.id == quizz_id).first()
    if quizz is None:
        abort(404, "Quizz not found")

    if len(user_answers) != len(quizz.picture_questions):
        abort(404, "Answer length doesn't match quizz length")

    if len(user_answers) == 0:
        abort(400, "User answers are empty. Cannot calculate score")

    score = sum(
        user_answer == question.awnser
        for user_answer, question in zip(user_answers, quizz.picture_questions)
    ) / len(user_answers)

    score_history: UserQuizzHistory | None = (
        g.session.query(UserQuizzHistory)
        .filter_by(user_id=user_id, quizz_id=quizz_id)
        .first()
    )

    if score_history:
        if score > score_history.best_score:
            score_history.best_score = score
        score_history.last_score = score
    else:
        score_history = UserQuizzHistory(
            user_id=user_id, quizz_id=quizz_id, best_score=score, last_score=score
        )
        g.session.add(score_history)

    try:
        g.session.commit()
    except Exception as e:
        g.session.rollback()
        logger.warning(f"SQLAlchemyException: {e}")
        return jsonify({"error": "Error while trying to save the score"}), 500

    return (
        jsonify(
            {
                "success": True,
                "object": {
                    "type": "QuizzHistory",
                    "quizz_history_id": score_history.id,
                    "quizz_name": quizz.quizz_name,
                    "last_score": score_history.last_score,
                    "best_score": score_history.best_score,
                    "user_id": user_id,
                },
                "computed_score": score,
            }
        ),
        200,
    )


@quizz_bp.route("/", methods=["DELETE"])
def delete_quizz():
    try:
        request_params = DeleteQuizzParams(**request.json)
    except ValidationError as e:
        return jsonify({"error": "Invalid input data", "details": str(e.errors())}), 400

    quizz_name = request_params.quizz_name
    quizz_id = request_params.quizz_id

    quizz_to_delete: Quizz | None = (
        g.session.query(Quizz)
        .filter(or_(Quizz.quizz_name == quizz_name, Quizz.id == quizz_id))
        .first()
    )
    if quizz_to_delete:
        try:
            manager = QuizzFileManager()
            g.session.delete(quizz_to_delete)
            g.session.commit()
            manager.delete_quizz_folder(quizz_to_delete)
            return (
                jsonify(
                    {
                        "success": True,
                        "object": {
                            "type": "Quizz",
                            "quizz_id": quizz_to_delete.id,
                            "quizz_name": quizz_to_delete.quizz_name,
                        },
                    }
                ),
                204,
            )
        except Exception as e:
            g.session.rollback()
            logger.warning(f"Error deleting quizz: {e}")
            return (jsonify({"error": "Error while deleting the quizz"}),)

    return jsonify({"error": "Quizz not found"}), 404
