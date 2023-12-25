from pydantic import BaseModel, EmailStr


class RegistrationModel(BaseModel):
    email: EmailStr
    password: str


class LoginModel(BaseModel):
    email: EmailStr
    password: str
