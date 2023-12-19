from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    ForeignKey,
)
from project.models.base import Base, BaseMixin
from dataclasses import dataclass
from sqlalchemy.orm import relationship


@dataclass()
class Quizz(Base, BaseMixin):
    __tablename__ = "quizz"

    quizz_name: str = Column(String(100), nullable=False)
    folder_name: str = Column(String(100), nullable=False, unique=True)
    available: str = Column(Boolean(), nullable=False, default=True)
    picture_questions: list["PictureQuestion"] = relationship(
        "PictureQuestion", back_populates="quizz", cascade="all, delete-orphan"
    )
    user_history = relationship("UserQuizzHistory", back_populates="quizz")

    @property
    def number_of_pictures(self):
        return len(self.pictures)


@dataclass()
class PictureQuestion(Base, BaseMixin):
    __tablename__ = "picture_question"
    filename: str = Column(String(200), nullable=False)
    old_filename: str = Column(String(200), nullable=False)
    awnser: int = Column(Integer, nullable=False)
    quizz_id: int = Column(Integer, ForeignKey("quizz.id"))
    quizz: Quizz = relationship("Quizz", back_populates="picture_questions")
