import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from common.config import Config
from routers import router_admin, router_score, router_stt


ENVIRONMENT: str = Config.get_env_value("ENVIRONMENT")


def create_app():
    ## 개발 환경이 아닐 때 API Docs 비활성화 처리
    if ENVIRONMENT == "production":
        app_instance = FastAPI(docs_url=None, redoc_url=None)
    else:
        app_instance = FastAPI()
    return app_instance


app: FastAPI = create_app()

if ENVIRONMENT == "production":
    origins = [
    ]
else:
    origins = [
        "http://localhost:9999",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    print(
        f"[Request]\
            \tmethod: {request.method}\
            \turl: {request.url}"
    )
    if request.method == "get":
        print(f"\tparams: {request.query_params}")
    else:
        print(f"\tbody: {request.body}")
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"[Response]\tstatus_code: {response.status_code}\tprocess_time: {process_time}")
    return response


app.include_router(router_stt.router)
app.include_router(router_score.router)


if __name__ == "__main__":
    if ENVIRONMENT == "production":
        log_level = "info"
    else:
        log_level = "debug"

    listen_port = int(Config.get_env_value(Config.make_env_value_key(key_name="APP_LISTEN_PORT", stage=ENVIRONMENT)))

    uvicorn.run(
        "app:app",
        port=listen_port,
        reload=False,
        log_level=log_level,
    )
