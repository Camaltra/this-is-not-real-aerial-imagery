from sqlalchemy import (
    Column,
    String,
    Boolean,
)
from project.models import Base, BaseMixin
from dataclasses import dataclass


@dataclass()
class User(Base, BaseMixin):
    __tablename__ = "users"

    username: str = Column(String(30), nullable=False)
    email: str = Column(String(254), unique=True, nullable=False)
    password: str = Column(String(60), nullable=False)
    active: bool = Column(Boolean(), default=True, nullable=False)
