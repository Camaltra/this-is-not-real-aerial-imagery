from sqlalchemy.orm import declarative_base
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func


Base = declarative_base()


@dataclass
class BaseMixin:
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), onupdate=func.now(), default=func.now())
