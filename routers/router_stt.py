from fastapi import UploadFile, APIRouter
from fastapi.responses import JSONResponse

from controllers import controller_stt as controller
from common.config import Config


_ROUTER_NAME: str = "stt"

router = APIRouter(
    prefix=f"/{_ROUTER_NAME}",
    tags=[_ROUTER_NAME.upper()],
)


@router.on_event("startup")
def startup_event():
    device_list = Config.get_env_value("STT_DEVICE_LIST").split(",")
    controller.initialize(device_list)


@router.on_event("shutdown")
def shutdown_event():
    controller.shutdown()


@router.get("/status")
def get_status():
    print(f"func: stt get_status")
    return JSONResponse(
        status_code=200,
        content={"code": 200},
    )


@router.post("/run")
def post_run(
    audio_file: UploadFile,
):
    print(f"func: stt post_run,\tparams: ({audio_file.filename})")

    try:
        result = controller.run_stt_by_uploadfile(audio_file=audio_file, is_refine=True)
        return JSONResponse(
            status_code=200,
            content=result,
        )
    except Exception as ex:
        print(ex)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": str(ex)},
        )
