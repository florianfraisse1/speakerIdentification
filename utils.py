import numpy as np
from scipy.signal import lfilter
import os
import json


def overwrite_config_with_env(path, obj):
    for item in obj.keys():
        if isinstance(obj[item], dict):
            obj[item] = overwrite_config_with_env(path + '_' + item, obj[item])
        else:
            cur_path = path + "_" + item
            if os.environ.get(cur_path.upper(), None) is not None:
                obj[item] = json.loads(os.environ[cur_path.upper()])
    return obj


def read_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    config = ''
    for line in lines:
        config += line.split("##", 1)[0]
    config = overwrite_config_with_env('ds_', json.loads(config))
    print("configuration set to " + str(config))
    return config


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)
    end_frame = int(max_sec * frames_per_sec)
    step_frame = int(step_sec * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1, -1], [1, - alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout
