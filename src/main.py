from fastapi import FastAPI

app = FastAPI()


@app.get(path = "/health")
def health_check():
    return{"status" : "ok"}

