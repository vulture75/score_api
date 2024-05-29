import json
from threading import Thread
from time import sleep

import boto3
from fastapi import APIRouter, BackgroundTasks
from fastapi.params import Body
from fastapi.responses import JSONResponse

from common.config import Config
from controllers import controller_score as controller


_ROUTER_NAME: str = "score"

router = APIRouter(
    prefix=f"/{_ROUTER_NAME}",
    tags=[_ROUTER_NAME.upper()],
)


IS_CONTINUE: bool = True

ENVIRONMENT: str = Config.get_env_value("ENVIRONMENT")

SQS_REGION_NAME: str = Config.get_env_value("AWS_SQS_REGION_NAME")
SQS_ACCOUNT_ID: str = Config.get_env_value("AWS_SQS_ACCOUNT_ID")
SQS_QUEUE_NAME: str = Config.get_env_value(Config.make_env_value_key("AWS_SQS_QUEUE_NAME", stage=ENVIRONMENT))

SQS_QUEUE_URL: str = f"https://sqs.{SQS_REGION_NAME}.amazonaws.com/{SQS_ACCOUNT_ID}/{SQS_QUEUE_NAME}"

SQS_CLIENT = None


def process_sqs_messages():
    global SQS_CLIENT, SQS_QUEUE_URL

    sleep(2)

    print("Listening for messages on %s" % SQS_QUEUE_URL)
    print("SQS Client: ", SQS_CLIENT)

    while IS_CONTINUE:
        response = SQS_CLIENT.receive_message(QueueUrl=SQS_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        # 받아온 메시지 처리
        messages = response.get("Messages", [])
        for message in messages:
            try:
                # 메시지 내용 처리 로직
                body = message["Body"]
                print("Received message:", body)
                body_dict = json.loads(body)

                stage = body_dict["stage"]
                service = body_dict["service"]

                if service == "class":
                    task_submit_id = body_dict["task_submit_id"]
                    controller.handler_make_report(task_submit_id=task_submit_id, service=service, stage=stage)
                else:
                    _id = body_dict["id"]
                    controller.handler_make_report(id=_id, service=service, stage=stage)

                # finally에서 처리하는 경우 다음 반복문에서 메시지가 중복으로 처리될 수 있음
                receipt_handle = message["ReceiptHandle"]
                SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)

            except Exception as ex:
                print("Error processing message:", ex)
                import traceback

                tb = ex.__traceback__
                tb_list = traceback.format_tb(tb)
                for tb_line in tb_list:
                    print(tb_line)


@router.on_event("startup")
def startup_event():
    global SQS_CLIENT, SQS_REGION_NAME
    controller.initialize()
    SQS_CLIENT = boto3.client(
        service_name="sqs",
        region_name=SQS_REGION_NAME,
        aws_access_key_id=Config.get_env_value("AWS_SQS_ACCESS_KEY"),
        aws_secret_access_key=Config.get_env_value("AWS_SQS_SECRET_KEY"),
    )
    Thread(target=process_sqs_messages).start()


@router.on_event("shutdown")
def shutdown_event():
    global IS_CONTINUE
    IS_CONTINUE = False


# @router.post(path="/make_report")
def make_report(
    task_submit_id: str = Body(),
    service: str = Body(),
    stage: str = Body(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    print(f"")
    print(
        f"task_submit_id: {task_submit_id}\
            service: {service}\
            stage: {stage}"
    )

    background_tasks.add_task(controller.handler_make_report, task_submit_id, service, stage)

    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
        },
    )
