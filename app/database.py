from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. The Database URL: This tells SQLAlchemy to create a local file named "investor.db"
SQLALCHEMY_DATABASE_URL = "sqlite:///./investor.db"

# 2. The Engine: This is the actual engine that manages the connection to SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} # This is a specific requirement for SQLite in FastAPI
)

# 3. The SessionLocal: A temporary workspace for our database operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. The Base: The master blueprint class that all our future database tables will inherit from
Base = declarative_base()

# 5. Dependency Function: We will use this in FastAPI to open a DB session and close it when done
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()