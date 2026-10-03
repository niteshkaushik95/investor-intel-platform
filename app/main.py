from fastapi import FastAPI
from .database import engine, Base    # <-- NEW
from . import models                  # <-- NEW

# Tell SQLAlchemy to create the tables in the database if they don't exist yet
Base.metadata.create_all(bind=engine) # <-- NEW

# 1. Create the FastAPI instance (the application itself)
app = FastAPI(title="Investor Intelligence API")

# 2. Define a "Route" or "Endpoint"
# The @ is a decorator. It tells FastAPI: "Whenever someone visits the root URL ('/'), run the function directly below it."
@app.get("/")
def health_check():
    # 3. Return a Python dictionary. FastAPI will automatically convert this to JSON!
    return {"status": "healthy", "message": "API is running!"}

@app.get("/about")
def about():
    return {"project": "Investor Intelligence Platform", "version": "1.0"}