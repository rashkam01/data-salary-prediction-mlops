FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv


COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_salary.py","model.xgb", "preprocessor.b","./" ]

EXPOSE 9696 

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696","predict_salary:app"]