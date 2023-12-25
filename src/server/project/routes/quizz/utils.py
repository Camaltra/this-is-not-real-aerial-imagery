from project.models.quizz import Quizz
import random
from project.connectors.s3 import S3Manager
from project.routes.quizz.constant import (
    BUCKET_NAME,
    DEEPFAKE_IMAGE_FOLDER,
    REAL_IMAGE_FOLDER,
    REAL_IMAGES_COUNT,
    MAX_REAL_IMAGES,
)
from project.routes.quizz.exceptions import InsufficientImagesError
import logging


class QuizzFileManager:
    def __init__(self):
        self.s3_manager = S3Manager(BUCKET_NAME)
        self.logger = logging.getLogger("QuizzFileManager")
        self.logger.setLevel(logging.WARNING)

    def build_quizz_folder(self, quizz: Quizz):
        folder_path = f"{quizz.folder_name}/"

        try:
            self.s3_manager.create_s3_folder(folder_path)

            for picture_question in quizz.picture_questions:
                self.s3_manager.move_file_within_bucket(
                    picture_question.old_filename, picture_question.filename
                )

            self.logger.info(f"Quizz folder '{folder_path}' successfully built.")
        except Exception as e:
            self.logger.error(f"Error building quizz folder: {e}")

    def delete_quizz_folder(self, quizz: Quizz):
        folder_path = f"{quizz.folder_name}/"

        try:
            for picture_question in quizz.picture_questions:
                self.s3_manager.move_file_within_bucket(
                    picture_question.filename, picture_question.old_filename
                )

            self.s3_manager.delete_folder(folder_path)

            self.logger.info(f"Quizz folder '{folder_path}' successfully deleted.")
        except Exception as e:
            self.logger.error(f"Error deleting quizz folder: {e}")


def build_quizz_content(
    manager: QuizzFileManager, number_of_pics: int
) -> tuple[list[str], list[int]]:
    real_pics_number = random.randint(REAL_IMAGES_COUNT, MAX_REAL_IMAGES)
    fake_pics_number = number_of_pics - real_pics_number

    available_real_images = manager.s3_manager.list_files_from_bucket(REAL_IMAGE_FOLDER)
    if len(available_real_images) < real_pics_number:
        raise InsufficientImagesError("Not enough real images, please collect more")
    real_images = random.sample(available_real_images, real_pics_number)

    available_fake_images = manager.s3_manager.list_files_from_bucket(
        DEEPFAKE_IMAGE_FOLDER
    )
    if len(available_fake_images) < fake_pics_number:
        raise InsufficientImagesError("Not enough fake images, please generate more")
    fake_images = random.sample(available_fake_images, fake_pics_number)

    images = real_images + fake_images
    targets = [0 for _ in range(real_pics_number)] + [
        1 for _ in range(fake_pics_number)
    ]

    shuffled_data = list(zip(images, targets))
    random.shuffle(shuffled_data)

    return zip(*shuffled_data)
