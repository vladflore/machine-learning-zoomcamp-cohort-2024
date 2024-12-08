FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl 

RUN pip install keras-image-helper

COPY [ "lambda_function.py", "./" ]

CMD [ "lambda_function.lambda_handler" ]
