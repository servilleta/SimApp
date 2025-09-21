from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import settings

# Use DATABASE_URL from environment variables
DATABASE_URL = settings.DATABASE_URL

# Create engine with appropriate configuration for different databases
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
elif DATABASE_URL.startswith("postgresql"):
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections after 5 minutes
        echo=False           # Set to True for SQL debugging
    )
else:
    # Fallback configuration
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Import all models so they're registered with SQLAlchemy
# This ensures Alembic can detect them for migrations
# Models are automatically discovered when imported elsewhere

# Function to create database tables (optional, can also use Alembic)
# Call this from main.py on startup if not using Alembic for initial creation
def create_db_tables():
    Base.metadata.create_all(bind=engine)
    print(f"Database tables created using: {DATABASE_URL[:50]}...") 