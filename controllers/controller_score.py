import sys
import os
import math
import random as rd
from tempfile import TemporaryDirectory
from time import sleep
from fastapi import UploadFile
import requests as reqs
from requests import Response
import numpy as np
import librosa as lrs
import pandas as pd
import pymysql
from datetime import datetime

IS_MAIN = __name__ == "__main__"
if IS_MAIN:
    sys.path.append("./")

from controllers import controller_stt as STT

from engine.AnswerAligner import AnswerAligner
from engine.ScoreCalculator import ScoreCalculator
from engine.VideoStat import VideoStat
from common.config import Config
from common import util
from common.util import S3, Json, ReportUtil
from common.const import ReportResultStatus, ScoreRatio

from core import intonation as IntonationCore


S3_CLIENT = None
BUCKET_NAME: str = None
VIDEO_STAT_INSTANCE: VideoStat = None

LOCAL_S3_SYNC_DIR = Config.get_env_value("LOCAL_S3_SYNC_DIR")
REPORT_JSON_DIR = f"{LOCAL_S3_SYNC_DIR}/2dub-class/report_v2/students"
SCORE_RATIO: ScoreRatio = ScoreRatio(recognition=0.4, speed=0.25, intonation=0.35)


def initialize():
    global S3_CLIENT, STT_PROCESSOR_INSTANCE, VIDEO_STAT_INSTANCE, STT_PROC_FOR_ROUTE

    S3_CLIENT = S3.init_s3_client()

    def report_file_iterator():
        dir_path = REPORT_JSON_DIR
        all_files = util.search_dir(dir_path=dir_path, target_ext=["json"])
        for file_path in all_files:
            yield Json.load(file_path)

    VIDEO_STAT_INSTANCE = VideoStat(
        report_iterator=report_file_iterator, matrix_length=20, refresh_rate=(0, 30, 0), ris_weight=(5, 3, 4), grade=6
    )

    pass


def handler_make_report(service, stage, **kwargs):
    global BUCKET_NAME
    try:
        temp_dir = TemporaryDirectory()

        BUCKET_NAME = Config.get_env_value(Config.make_env_value_key("S3_BUCKET_NAME", service, stage))

        match service:
            case "class":
                base_data = get_task_data(task_submit_id=kwargs["task_submit_id"], service=service, stage=stage)
            case _:  # default
                base_data = get_base_data(_id=kwargs["id"], service=service, stage=stage)

        language = base_data["video"]["language"]
        print(f"Video Language: {language}")
        score_dict = calculate_score(task_data=base_data, temp_dir_path=temp_dir.name, stage=stage, language=language)
        report_json = make_report_json(base_data, score_dict)

        assert ReportUtil.verify_report_file(report_json, service)

        Json.save(report_json, "report_result.json")
        upload_url = upload_and_save_report(report_json, service, **kwargs)

        print(report_json)

        if service == "class":
            call_complete_api(
                is_error=False,
                service=service,
                stage=stage,
                score=report_json["video"]["score"],
                report_url=upload_url,
                task_submit_id=kwargs["task_submit_id"],
            )
        else:
            call_complete_api(
                is_error=False,
                service=service,
                stage=stage,
                score=report_json["video"]["score"],
                report_url=upload_url,
                id=kwargs["id"],
            )
    except Exception as ex:
        print(ex)
        import traceback

        tb = ex.__traceback__
        tb_list = traceback.format_tb(tb)
        for tb_line in tb_list:
            print(tb_line)

        if service == "class":
            call_complete_api(task_submit_id=kwargs["task_submit_id"], is_error=True, service=service, stage=stage)
        else:
            call_complete_api(id=kwargs["id"], is_error=True, service=service, stage=stage)

        raise Exception("Error in make_report")
    finally:
        temp_dir.cleanup()
    pass


def run_stt(audio_file: UploadFile, language: str) -> dict:
    temp_dir = TemporaryDirectory()
    temp_file_path = f"{temp_dir.name}/{audio_file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(audio_file.file.read())

    res = STT_PROC_FOR_ROUTE.run(file_url=temp_file_path)
    return res


def verify_task_data(task_data):
    scripts = task_data["scripts"]
    file_url_none_count = len(list(filter(lambda x: x["file_url"] is None, scripts)))
    if file_url_none_count > 0:
        return 1
    return 0


def get_task_data(
    service,
    stage,
    **kwargs,
):
    req_url = make_api_url_for_get_base_data(service, stage, task_submit_id=kwargs["task_submit_id"])

    response: Response = reqs.get(req_url, headers=make_req_header_for_call_api(service, stage))
    res_data = response.json()["data"]
    if verify_task_data(res_data) != 0:
        raise Exception("Invalid task data")
    return res_data


def get_base_data(_id, service, stage):
    req_url = make_api_url_for_get_base_data(service, stage, id=_id)
    print(req_url)
    response: Response = reqs.get(req_url)  # , headers=make_req_header_for_call_api(service, stage))
    print(response.json())
    res_data = response.json()["data"]
    if verify_task_data(res_data) != 0:
        raise Exception("Invalid task data")
    return res_data


def calculate_score(
    task_data: dict,
    temp_dir_path: str,
    stage: str = "test",
    org_stt_result_dir: str = None,
    make_org_stt_file_name=None,
    language: str = "en",
):
    # score based STT
    print("Org. STT")
    org_result = get_org_stt_result(
        task_data=task_data,
        save_dir=temp_dir_path,
        stage=stage,
        language=language,
        org_stt_result_dir=org_stt_result_dir,
        make_org_stt_file_name=make_org_stt_file_name,
    )

    print("User STT")
    user_result = get_user_stt_result(task_data=task_data, save_dir=temp_dir_path, language=language)

    print("Score")
    stt_score_dict = ScoreCalculator.run(org_result, user_result)

    # score based Intonation
    into_score_dict = get_intonation_score(
        save_dir=temp_dir_path,
        video_idx=task_data["video"]["idx"],
        line_count=task_data["video"]["line_count"],
        script_url_list=list(map(lambda x: x["file_url"], task_data["scripts"])),
    )

    print("Make Final JSON")
    script_score_dict = make_script_score_dict(stt_score_dict, into_score_dict)
    score_dict = calculate_video_score(script_score_dict)

    return score_dict


def make_report_json(task_data: dict, score_dict: dict):
    # make info for report (stats, history, ...)
    score_info = add_stats_and_history(task_data, score_dict)

    # merge data and score
    res_dict = merge_data_and_score(task_data, score_info)
    return res_dict


def upload_report_to_s3(report_file_path, save_base_path):
    print("Upload report to S3")
    S3.upload_file_to_s3(src_file_path=report_file_path, bucket_name=BUCKET_NAME, s3_save_path=save_base_path)
    s3_full_url = f"https://{BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/{save_base_path}"
    return s3_full_url


def call_complete_api(
    is_error: bool,
    service: str,
    stage: str,
    score: dict = None,
    report_url: str = None,
    **kwargs,
):
    import json as js

    print("Call patch API")
    if service == "class":
        req_url = make_api_url_for_get_base_data(
            service, stage, added_path="/speaking-report", task_submit_id=kwargs["task_submit_id"]
        )
    else:
        req_url = make_api_url_for_get_base_data(
            service, stage, added_path="/report", add_query_params=False, id=kwargs["id"]
        )

    headers = make_req_header_for_call_api(service, stage)

    if is_error:
        params = {
            "result_status": ReportResultStatus.ERROR.value,
            "report_score": 0,
            "recognized_score": 0,
            "speed_score": 0,
            "intonation_score": 0,
            "stability_score": 0,
        }
    else:
        params = {
            "result_status": ReportResultStatus.COMPLETE.value,
            "file_url": report_url,
            "report_score": score["total"],
            "recognized_score": score["ris"]["recognition"]["user"],
            "speed_score": score["ris"]["speed"]["user"],
            "intonation_score": score["ris"]["intonation"]["user"],
            "stability_score": score["ris"]["stability"],
        }
    if service != "class":
        params["service_platform"] = service.strip()

    print("\nURL: ", req_url)
    print("\nHeaders: ", headers)
    print("\nParams: ", params)
    if service == "class":
        response = reqs.patch(req_url, data=params, headers=headers)
    else:
        response = reqs.patch(req_url, data=params, headers=headers)
    print(response.text)


####################### [Sub Func.] ##########################


def make_api_url_for_get_base_data(
    service: str, stage: str, added_path: str = None, add_query_params: bool = True, **kwargs
):
    if service == "class":
        url_prefix = Config.get_env_value(Config.make_env_value_key("API_URL", service, stage))
        url_path = f"api/v3/*****/***/***/{kwargs['task_submit_id']}"
        res_url = f"{url_prefix}/{url_path}"
        if added_path:
            res_url += f"{added_path}"
    else:
        url_prefix = Config.get_env_value("COMMON_API_URL")
        url_path = f"api/v3/*****/***/***/{kwargs['id']}"
        _stage: str = "prod" if stage == "qa" else stage
        res_url = f"{url_prefix}/{url_path}"
        if added_path:
            res_url += f"{added_path}"
        if add_query_params:
            res_url += f"?service_platform={service}"
    return res_url


def make_req_header_for_call_api(service: str, stage: str):
    api_key = Config.get_env_value(Config.make_env_value_key("API_KEY", service, stage))
    if service == "class":
        return {"key": api_key}
    else:
        return {}


def get_stats_info(task_data, score_dict):
    video_idx = task_data["video"]["idx"]

    stat_result = VIDEO_STAT_INSTANCE.run(
        video_idx=video_idx,
        total_score=score_dict["video"]["total"],
    )

    dominant_matrix_dict = get_dominant_matrix(
        task_data["scripts"], score_dict["scripts"], score_dict["video"]["ris"]["intonation"]
    )
    stat_result["avg_intonation_ratio"] = dominant_matrix_dict

    return stat_result


def get_dominant_matrix(info_dict, score_dict, avg_intonation_score):
    dominant_org_matrix, min_diff = [], 1.0

    for i, script in enumerate(score_dict):
        diff = abs(avg_intonation_score - script["score"]["intonation"])
        if diff < min_diff:
            dominant_org_matrix = info_dict[i]["matrix"]
            min_diff = diff

    offset = math.ceil(len(dominant_org_matrix) * ((1.0 - avg_intonation_score) / 2))

    dominant_user_matrix = []
    for i in range(offset, len(dominant_org_matrix)):
        dominant_user_matrix.append(
            dominant_org_matrix[i] * (1.0 + (rd.randrange(0, 3) - 1) * (rd.randrange(1, 4) * 0.1))
        )
    dominant_user_matrix.extend([0 for _ in range(offset)])

    res_mat_info = {
        "ratio_value": avg_intonation_score,
        "org_matrix": dominant_org_matrix,
        "user_matrix": dominant_user_matrix,
    }

    return res_mat_info


####################### remove for public ##########################


def save_report_to_file(report_json, s3_save_path, save_path_prefix=f"{LOCAL_S3_SYNC_DIR}"):
    print("Save report to file")
    full_file_path = f"{save_path_prefix}/{BUCKET_NAME}/{s3_save_path}"
    util.create_directory(os.path.dirname(full_file_path))
    Json.save(report_json, full_file_path)
    return full_file_path


if __name__ == "__main__":

    initialize()

    handler_make_report(task_submit_id="abcdefg1234567890", service="class", stage="test")
