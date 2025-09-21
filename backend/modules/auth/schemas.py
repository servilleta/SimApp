from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, pattern="^[a-zA-Z0-9_ ]+$")
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_admin: Optional[bool] = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=100)
    password_confirm: str

class User(UserBase):
    id: int
    disabled: Optional[bool] = None

    class Config:
        from_attributes = True

class UserInDB(User):
    hashed_password: str

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_admin: Optional[bool] = None
    disabled: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)

class UserCreateAdmin(UserBase):
    password: str = Field(..., min_length=8, max_length=100) 