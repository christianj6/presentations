FROM ghcr.io/predibase/lorax:main

COPY code/middleware.py .
RUN pip install fastapi httpx uvicorn

# in a production setting your middleware can be another system component
# here we just add another fastapi in front of LoRAX, inside of the same container
ENTRYPOINT ["python", "middleware.py"]

