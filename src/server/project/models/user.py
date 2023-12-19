from sqlalchemy import (
    Column,
    String,
    Boolean,
)
from project.models.base import Base, BaseMixin
from dataclasses import dataclass
from sqlalchemy.orm import relationship


@dataclass()
class User(Base, BaseMixin):
    __tablename__ = "user"

    username: str = Column(String(30), nullable=False)
    email: str = Column(String(254), unique=True, nullable=False)
    password: str = Column(String(60), nullable=False)
    active: bool = Column(Boolean(), default=True, nullable=False)
    admin: bool = Column(Boolean(), default=False, nullable=False)

    quizz_history = relationship("UserQuizzHistory", back_populates="user")
