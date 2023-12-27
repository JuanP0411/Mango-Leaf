from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routes import router


app = FastAPI()
origins = ["http://localhost", "http://localhost:5173"]  # List of allowed origins (you can customize this)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":

    uvicorn.run(app=app, host="127.0.0.1", port=5555, log_level= "debug")