from pydantic import BaseModel, EmailStr


class GetUserParams(BaseModel):
    user_id: str


class UpdateUserRequest(BaseModel):
    username: str = None
    email: EmailStr = None
