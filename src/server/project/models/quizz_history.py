from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from dataclasses import dataclass
from project.models.base import Base, BaseMixin


@dataclass()
class UserQuizzHistory(Base, BaseMixin):
    __tablename__ = "user_quizz_history"

    user_id: int = Column(Integer, ForeignKey("user.id"), nullable=False)
    quizz_id: int = Column(Integer, ForeignKey("quizz.id"), nullable=False)
    score: float = Column(Float, nullable=False)
    user = relationship("User", back_populates="quizz_history")
    quizz = relationship("Quizz", back_populates="user_history")
