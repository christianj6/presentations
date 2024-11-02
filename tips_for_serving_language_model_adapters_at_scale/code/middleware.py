"""
This module is intended to accompany the
included Dockerfile and serve as a demonstration
of how one could "wrap" a LoRAX server in order to gain
additional control over inferencing or add any additional
middleware logic such as routing or even generation structuring.
"""
import subprocess
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # we can pass any lorax-launcher parameters necessary,
    # or even include the environment from the container itself,
    # when calling Popen, if this is more convenient for deployment
    process = subprocess.Popen(
        [
            "lorax-launcher",
            "--quantize",
            "bitsandbytes-fp4",
            "--model-id",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
    )

    try:
        app.state.lorax_process = process
        yield

    finally:
        process.terminate()
        process.wait()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: Request):
    # in this example, we simply forward the request,
    # but one would likely want to expose a different interface to
    # upstream clients and take advantage of the opportunity
    # for pre/post -processing of requests
    body = await request.json()

    # todo: custom logic

    timeout = httpx.Timeout(10.0, read=None)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://0.0.0.0:80/generate",  # note that we target the LoRAX server port
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


if __name__ == "__main__":
    import uvicorn

    # note that we expose a different port for the "middleware" server
    # this port must be exposed when running the container
    uvicorn.run(app, host="0.0.0.0", port=8000)
