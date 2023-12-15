from datetime import datetime
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
from sqlalchemy.orm import declarative_base, relationship, Mapped
from sqlalchemy.sql import func
from typing import List
from dataclasses import dataclass
from project.models import Base, BaseMixin


class User(Base, BaseMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(128), unique=True, nullable=False)
    active = Column(Boolean(), default=True, nullable=False)

    def __init__(self, email):
        self.email = email
