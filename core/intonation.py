import os
import numpy as np
import librosa as lrs
from scipy.signal import wiener
from scipy.stats import spearmanr


def calculate_score(
    org_voice_path, user_voice_path, with_denoising=True, feature_type="mfcc", audio_sample_rate=16000, user_save_path=None, **kwargs
):
    step_progress = 1    
    try:
        print(f"[Step {step_progress}] load voice data...")
        org_voice_arr, _ = lrs.load(org_voice_path, sr=audio_sample_rate)
        user_voice_arr, _ = lrs.load(user_voice_path, sr=audio_sample_rate)
        
        step_progress += 1
        if with_denoising:
            print(f"[Step {step_progress}] apply denoising with Wiener Filter...")
            org_voice_arr_filtered = _apply_wiener_filter(voice_arr=org_voice_arr)
            user_voice_arr_filtered = _apply_wiener_filter(voice_arr=user_voice_arr)
        else:
            print(f"[Step {step_progress}] skip denoising...")

        step_progress += 1
        print(f"[Step {step_progress}] extract voice feature...")
        audios = [(org_voice_arr_filtered, org_voice_arr), (user_voice_arr_filtered, user_voice_arr)]
        features = []
        for audio_tuple in audios:
            try:
                feature = _extract_voice_feature(
                    voice_arr=audio_tuple[0],
                    feature_type=feature_type,
                )
            except lrs.ParameterError:
                feature = _extract_voice_feature(
                    voice_arr=audio_tuple[1],
                    feature_type=feature_type,
                )
            features.append(feature)
        feature_org, feature_user = features[0], features[1]
        
        step_progress += 1
        print(f"[Step {step_progress}] align with DTW...")
        dtw_distances, dtw_path = _run_dtw(feature1=feature_org.T, feature2=feature_user.T)

        step_progress += 1
        print(f"[Step {step_progress}] calculate correlation...")
        corr_score = _calculate_correlation(dtw_path, feature_org, feature_user)

        step_progress += 1
        print(f"[Step {step_progress}] caculate length score...")
        len_score = _calculate_length_score(org_voice_arr, user_voice_arr)

        step_progress += 1
        print(f"[Step {step_progress}] caculate total score...")
        total_score = _calculate_total_score(corr_score, len_score)

        print(f"[Done] Intonation Score Result >> {total_score}")
        print(f"\tCorrelation Score >> {corr_score}")
        print(f"\tLength Score      >> {len_score}")

        return round(total_score, 4)

    except Exception as ex:
        print(f"[Error] error in step {step_progress} of separating...")
        print(f"\tErrorMsg >> {ex.__str__()}")

        raise ex


################################# Sub Functions #################################


def _apply_wiener_filter(voice_arr):
    return wiener(voice_arr)


def _generate_gaussian_noise(length, mean=0, std=0.01):
    """
    가우시안 노이즈 생성 함수
    :param length: 노이즈의 길이 (샘플 수)
    :param mean: 노이즈의 평균 (기본값: 0)
    :param std: 노이즈의 표준편차 (기본값: 0.01)
    :return: 생성된 가우시안 노이즈
    """
    noise = np.random.normal(loc=mean, scale=std, size=length)
    return noise


def _calculate_snr(signal, noise):
    """
    신호 대 잡음 비율(SNR) 계산 함수
    """
    signal_power = np.sum(signal**2) / len(signal)
    noise_power = np.sum(noise**2) / len(noise)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def _vad_snr(audio, snr_threshold=10, frame_length=2048, hop_length=512):
    """
    SNR 기반 음성 활동 감지(VAD) 함수
    :param audio: 오디오 PCM 데이터
    :param snr_threshold: 음성 활동으로 감지할 최소 SNR 임계값 (기본값: 20 dB)
    :param frame_length: 오디오 프레임의 길이 (기본값: 2048)
    :param hop_length: 오디오 프레임 간의 건너뛰기 길이 (기본값: 512)
    :return: 음성 활동으로 감지된 시간 구간 (시작시간, 끝시간) 리스트, 전체 프레임 수
    """
    frames = lrs.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
    vad_segments = []
    snr_arr = []

    for i, frame in enumerate(frames):
        snr = _calculate_snr(frame, _generate_gaussian_noise(len(frame)))
        snr_arr.append(snr)
        if snr >= snr_threshold:
            vad_segments.append(i)

    if len(vad_segments) < 1:
        return (None, None), len(frames)
    return (vad_segments[0], vad_segments[-1]), len(frames)


def _extract_voice_feature(voice_arr, feature_type="mfcc", save_path=None):
    file_full_path = f"{save_path}_{feature_type}.npy"
    if save_path is None or not os.path.exists(save_path):
        match feature_type:
            case "chroma":
                feature = lrs.feature.chroma_stft(y=voice_arr, sr=16000, n_chroma=13)
            case _:  # default and mfcc
                feature = lrs.feature.mfcc(y=voice_arr, sr=16000, n_mfcc=13)
        np.save(file_full_path, feature)
    else:
        feature = np.load(file_full_path)
    return feature


def _run_dtw(feature1, feature2):
    n, m = len(feature1), len(feature2)
    dtw_distances = np.zeros((n, m))
    dtw_path = np.zeros((n, m), dtype=int)

    for i in range(n):
        for j in range(m):
            cost = np.linalg.norm(feature1[i] - feature2[j])  # 거리 계산
            if i == 0 and j == 0:
                dtw_distances[i, j] = cost
            elif i == 0:
                dtw_distances[i, j] = cost + dtw_distances[i, j - 1]
            elif j == 0:
                dtw_distances[i, j] = cost + dtw_distances[i - 1, j]
            else:
                min_distance = min(
                    dtw_distances[i - 1, j],  # 위쪽
                    dtw_distances[i, j - 1],  # 왼쪽
                    dtw_distances[i - 1, j - 1],  # 대각선
                )
                dtw_distances[i, j] = cost + min_distance
                dtw_path[i, j] = (
                    np.argmin(
                        [
                            dtw_distances[i - 1, j],
                            dtw_distances[i, j - 1],
                            dtw_distances[i - 1, j - 1],
                        ]
                    )
                    + 1
                )

    # 최적 경로 계산
    opt_path = []
    opt_distances = []
    i, j = n - 1, m - 1
    opt_path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_idx = np.argmin(
                [
                    dtw_distances[i - 1, j],
                    dtw_distances[i, j - 1],
                    dtw_distances[i - 1, j - 1],
                ]
            )
            if min_idx == 0:
                i -= 1
            elif min_idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        opt_path.append((i, j))
        opt_distances.append(dtw_distances[i, j])

    res_path = np.array(opt_path[::-1])
    res_distances = np.array(opt_distances[::-1])

    return res_distances, res_path


############ remove for public ############

def _multi_dim_feature_to_single_value(mfcc, coef_num=5, weight_multiplier=1.5, first_weight=1.0):
    return [0]


def _calculate_correlation(dtw_path, org_mfcc, user_mfcc):
    return 0


def _calculate_length_ratio(org_length, user_length):
    return 1


def _crop_voice(audio_arr):
    return audio_arr


def _calculate_length_score(org_arr, user_arr):
    return 0


def _calculate_total_score(corr_score, len_score):
    return 1
