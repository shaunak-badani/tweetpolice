from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique = True)

    def __repr__(self):
        return f"<User id={self.id} name={self.name}>"
    
    class Config:
        orm_mode = True


class UserResponse(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True  # This tells Pydantic to treat the SQLAlchemy model as a dict