FROM python:3.11-slim

#setting the directory
WORKDIR /app

#copy requirements COPY [file on my machine] [where in the image]
COPY ./requirements.txt ./requirements.txt

# installing the dependencies from requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# copy the app from local to the image 

COPY ./src ./src

# the startup command 

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
