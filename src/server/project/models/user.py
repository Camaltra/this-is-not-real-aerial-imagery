from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
)
from project.models import Base


class User(Base):
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True)
    email: str = Column(String(128), unique=True, nullable=False)
    active: bool = Column(Boolean(), default=True, nullable=False)

    def __init__(self, email):
        self.email = email
