FROM svizor/zoomcamp-model:3.11.5-slim 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv

RUN pipenv install --deploy --system

COPY ["*.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
