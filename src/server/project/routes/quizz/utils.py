from flask import abort
from project.models.quizz import Quizz
import random
from project.connectors.s3 import S3Manager
from project.routes.quizz.constant import (
    BUCKET_NAME,
    DEEPFAKE_IMAGE_FOLDER,
    REAL_IMAGE_FOLDER,
)


class QuizzFileManager:
    def __init__(self):
        self.s3_manager = S3Manager(BUCKET_NAME)

    def build_quizz_folder(self, quizz: Quizz):
        self.s3_manager.create_s3_folder(f"{quizz.folder_name}/")
        for picture_question in quizz.picture_questions:
            self.s3_manager.move_file_within_bucket(
                picture_question.old_filename, picture_question.filename
            )

    def delete_quizz_folder(self, quizz: Quizz):
        for picture_question in quizz.picture_questions:
            self.s3_manager.move_file_within_bucket(
                picture_question.filename, picture_question.old_filename
            )
        self.s3_manager.delete_folder(f"{quizz.folder_name}/")


def build_quizz_content(
    manager: QuizzFileManager, number_of_pics: int
) -> tuple[list, list]:
    real_pics_number = random.randint(1, 4)
    fake_pics_number = number_of_pics - real_pics_number

    available_real_images = manager.s3_manager.list_files_from_bucket(REAL_IMAGE_FOLDER)
    if len(available_real_images) < real_pics_number:
        abort(403, "No enought real image, pls collect more")
    real_images = random.sample(available_real_images, real_pics_number)

    available_fake_images = manager.s3_manager.list_files_from_bucket(
        DEEPFAKE_IMAGE_FOLDER
    )
    if len(available_fake_images) < fake_pics_number:
        abort(403, "No enought faker image, pls generate more")
    fake_images = random.sample(available_fake_images, fake_pics_number)

    images = real_images + fake_images
    targets = [0 for _ in range(real_pics_number)] + [
        1 for _ in range(fake_pics_number)
    ]

    combined = list(zip(images, targets))

    random.shuffle(combined)

    return zip(*combined)
