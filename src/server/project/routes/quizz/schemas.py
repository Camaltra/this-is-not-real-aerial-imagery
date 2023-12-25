from pydantic import BaseModel


class QuizzPictureRequest(BaseModel):
    quizz_id: int
    question_ix: int


class QuizzCreateRequest(BaseModel):
    quizz_name: str
    number_of_pics: int = 5


class ComputeScoresAndSaveParams(BaseModel):
    user_id: int
    user_answers: list[str]
    quizz_id: int


class DeleteQuizzParams(BaseModel):
    quizz_name: str = None
    quizz_id: int = None
