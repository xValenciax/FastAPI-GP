from fastapi import FastAPI
from Routes.router import router as routes

app = FastAPI()

app.include_router(routes)
