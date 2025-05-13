from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from schema import Base, User

DATABASE_URL = "postgresql://postgres:password@localhost:5432/app_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DataFetcher:

    @classmethod
    def get_users(cls, db):
        users = db.query(User).all()
        return users
    
    @classmethod
    def get_user(cls, db, user_id):
        user = db.query(User).filter(User.id == user_id).first()
        return user