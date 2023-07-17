FROM public.ecr.aws/lambda/python:3.10
WORKDIR ${LAMBDA_TASK_ROOT}
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}
COPY . .
ENV JOBLIB_MULTIPROCESSING=0
RUN pip install --no-cache-dir -r requirements.txt
CMD ["src.app.main.handler"]