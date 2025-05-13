from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.exc import IntegrityError
from schema import User, Base
import json

# Connect to your local Docker PostgreSQL
DATABASE_URL = "postgresql://postgres:password@localhost:5432/app_db"
engine = create_engine(DATABASE_URL)

# Create tables and insert rows
def load_all_data():
    with open("initial_data.json") as f:
        users = json.load(f)
    
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        for user in users:
            session.add(User(name=user["name"]))
            try:
                session.commit()
            except IntegrityError:
                print(f"User : {user['name']} already exists")
                session.rollback()
        
        print("Operation complete")

if __name__ == "__main__":
    load_all_data()