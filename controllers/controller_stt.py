from copy import deepcopy
from enum import Enum
import os
import shutil
from multiprocessing import Pipe, Process
import sys
from time import sleep
from queue import Queue
from threading import Semaphore
import traceback

from fastapi import UploadFile
from engine.STTProcessor import STTProcessor


class EVENT_STRING(Enum):
    READY = "ready"
    EXIT = "exit"
    SUCCESS = "success"
    FAIL = "fail"


RESOURCE_LIST: Queue[STTProcessor] = None
SEMAPHORE: Semaphore = None
SUB_PROCESS_LIST: list[Process] = []


def initialize(_device_list):
    global RESOURCE_LIST, SEMAPHORE, SUB_PROCESS_LIST

    device_count = len(_device_list)
    SEMAPHORE = Semaphore(device_count)
    boot_resource_list = []
    RESOURCE_LIST = Queue(device_count)

    for _device_string in _device_list:
        device, device_index = _device_string.split(":")
        print(f"device: {device}, device_index: {device_index}")

        parent_conn, child_conn = Pipe()
        RESOURCE_LIST.put(parent_conn)

        proc = Process(target=_looper, args=(device, int(device_index), child_conn))
        proc.start()

        SUB_PROCESS_LIST.append(proc)

    for conn in boot_resource_list:
        res = conn.recv()
        if res != EVENT_STRING.READY.value:
            raise Exception("STTProcessor initialize failed")
        else:
            RESOURCE_LIST.put(conn)

    pass


def shutdown():
    global SUB_PROCESS_LIST

    while RESOURCE_LIST.empty() is False:
        RESOURCE_LIST.get().send(EVENT_STRING.EXIT.value)

    for proc in SUB_PROCESS_LIST:
        proc.join()


def run_stt_by_uploadfile(audio_file: UploadFile, language: str = None, is_refine: bool = True):
    temp_dir = _make_temp_dir()
    try:
        file_path = _save_audio_file(audio_file, save_dir=temp_dir)
        res = run_stt(file_path=file_path, language=language, is_refine=is_refine)
        _remove_temp_dir(temp_dir)
        return res
    except Exception as ex:
        print(ex, file=sys.stderr)
        traceback.print_exc()
        traceback.print_stack()
        # print(ex)
        raise ex


def run_stt(file_path: str | list[str], language: str = None, is_refine: bool = True):
    try:
        print(f"file_path: {file_path}")

        resource = _get_resource()
        res = _process(resource, file_path=file_path, language=language)
        print(res)
        try:
            if is_refine:
                res = _refine_stt_result(res)
        except Exception as ex:
            print(ex)
            pass
        return res

    except Exception as ex:
        print(ex, file=sys.stderr)
        traceback.print_exc()
        traceback.print_stack()
        # print(ex)
        raise ex

    finally:
        _release_resource(resource)


########################################################################################################################


def _make_temp_dir():
    _dir_path = f"/tmp/{os.urandom(24).hex()}"
    os.mkdir(_dir_path)
    return _dir_path


def _remove_temp_dir(_dir_path: str):
    shutil.rmtree(_dir_path)
    pass


def _save_audio_file(audio_file: UploadFile, save_dir: str):
    file_path = f"{save_dir}/{audio_file.filename[-30:]}"

    with open(file_path, "wb") as f:
        f.write(audio_file.file.read())

    return file_path


def _get_resource(_pooling_interval_sec: float = 0.5):
    global RESOURCE_LIST, SEMAPHORE

    SEMAPHORE.acquire()
    resource = None
    while resource is None:
        resource = RESOURCE_LIST.get()
        sleep(_pooling_interval_sec)

    return resource


def _release_resource(resource):
    global RESOURCE_LIST, SEMAPHORE

    RESOURCE_LIST.put(resource)
    SEMAPHORE.release()


def _looper(device: str, device_index: int, pipe: Pipe):
    # if device == "cuda":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    compute_type = "float16" if device == "cuda" else "int8"
    stt_proc = STTProcessor(
        device=device, device_index=device_index, batch_size=4, compute_type=compute_type, verbosity=1
    )

    print(f"Ready STT Looper: {device}:{device_index}, {compute_type}")

    pipe.send(EVENT_STRING.READY.value)

    while True:
        msg = pipe.recv()
        if msg == EVENT_STRING.EXIT.value:
            break
        if msg is None:
            continue
        try:
            (file_path, language) = msg
            res = stt_proc.run(file_url=file_path, language=language)
            pipe.send((EVENT_STRING.SUCCESS.value, res))
        except Exception as ex:
            print(ex)
            pipe.send((EVENT_STRING.FAIL.value, ex))


def _process(pipe: Pipe, file_path: str, language: str = None):
    pipe.send((file_path, language))
    res = pipe.recv()

    # Ready Message가 한번 더 날아오는 경우가 있어서 그에 대한 예외처리
    try:
        status, result = res
    except Exception as ex:
        if res == EVENT_STRING.READY.value:
            status, result = pipe.recv()
        else:
            raise ex

    if status == EVENT_STRING.SUCCESS.value:
        return result
    else:
        raise result


def _refine_stt_result(stt_result: dict):
    NO_TIMESTAMP_ERROR_RETRY_COUNT = 3
    FORCE_SPLIT_SINGLE_WORD_LENGTH = 1.5
    FORCE_SPLIT_GAP_BETWEEN_WORDS = 1

    old_segments = stt_result["segments"]

    # merge incomplete words

    for i, segment in enumerate(old_segments):
        complete_words_count = 0
        words = segment["words"]
        if len(words) == 1:
            if len(words[0].keys()) < 3:
                words[0]["start"] = segment["start"]
                words[0]["end"] = segment["end"]

        for word in words:
            if len(word.keys()) > 3:
                complete_words_count += 1

        if complete_words_count == 0:
            # find closest segment
            prev_segment = old_segments[i - 1]
            next_segment = old_segments[i + 1]
            gap_to_prev = segment["start"] - prev_segment["end"]
            gap_to_next = next_segment["start"] - segment["end"]

            if gap_to_prev < gap_to_next:
                for word in segment["words"]:
                    prev_segment["words"].append(word)
                prev_segment["end"] = segment["end"]
            else:
                for word in segment["words"]:
                    next_segment["words"].insert(0, word)
                next_segment["start"] = segment["start"]

            del old_segments[i]

    # split long words & large gap between words

    new_segments = []
    for segment in old_segments:
        new_words = []
        prev_end = -1
        for word in segment["words"]:
            try:
                word_length = word["end"] - word["start"]
            except KeyError:
                new_words.append(word)
                continue

            if prev_end == -1:
                prev_end = word["start"]

            if word_length > FORCE_SPLIT_SINGLE_WORD_LENGTH:
                new_words.append(word)
                new_segments.append({"words": new_words})
                new_words = []
            elif word["start"] - prev_end > FORCE_SPLIT_GAP_BETWEEN_WORDS:
                new_segments.append({"words": new_words})
                new_words = []
                new_words.append(word)
            else:
                new_words.append(word)

            prev_end = word["end"]

        if len(new_words) > 0:
            new_segments.append({"words": new_words})

    for segment in new_segments:
        if len(segment["words"]) == 0:
            continue

        # if len(segment["words"]) > 1:
        #     for word in segment["words"]:
        #         try:
        #             duration = word["end"] - word["start"]
        #             if duration > 2:
        #                 word["end"] = word["start"] + 2
        #         except KeyError:
        #             continue

        for i in range(NO_TIMESTAMP_ERROR_RETRY_COUNT):
            try:
                segment["start"] = segment["words"][i]["start"]
                break
            except KeyError:
                continue

        for i in range(NO_TIMESTAMP_ERROR_RETRY_COUNT):
            try:
                segment["end"] = segment["words"][-(i + 1)]["end"]
                break
            except KeyError:
                continue

        if stt_result["language"] in ["ja", "japanese"]:
            segment["text"] = "".join([word["word"] for word in segment["words"]])
        else:
            segment["text"] = " ".join([word["word"] for word in segment["words"]])

    new_result = deepcopy(stt_result)
    new_result["segments"] = new_segments
    return new_result
