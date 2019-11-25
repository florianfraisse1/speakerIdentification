import numpy as np
from scipy.spatial.distance import cdist
from model import vggvox_model
from utils import remove_dc_and_dither, normalize_frames, build_buckets, read_config
import sigproc
import speech_recognition as sr
import pickle

config = read_config('./resources/config.json')


def get_embedding(model, frames, max_sec):
    buckets = build_buckets(max_sec, config['bucketStep'], config['frameStep'])
    fft = abs(np.fft.fft(frames, n=config['numFFT']))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1] - rsize) / 2)
    signal = fft_norm[:, rstart:rstart + rsize]
    embedding = np.squeeze(model.predict(signal.reshape(1, *signal.shape, 1)))
    return embedding


def preprocess_signal(audio):
    signal = np.fromstring(audio, dtype=np.int16)
    signal = signal / 32768
    signal *= 2 ** 15
    signal = remove_dc_and_dither(signal, config['sampleRate'])
    signal = sigproc.preemphasis(signal, coeff=config['preemphasisAlpha'])
    frames = sigproc.framesig(signal, frame_len=config['frameLength'] * config['sampleRate'], frame_step=config['frameStep'] * config['sampleRate'], winfunc=np.hamming)
    return frames


def enrollment(name, embs_dict, model):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:  # mention source it will be either Microphone or audio files.
        print("Enrollment phase : Speak to enroll yourself")
        audio = r.listen(source)  # listen to the source

    frames = preprocess_signal(audio.frame_data)

    new_embedding = get_embedding(model, frames, config['maxSeconds'])
    embs_dict[name] = new_embedding
    with open(config['enrollmentFile'], 'wb') as f:
        pickle.dump(embs_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Enrollment finished, Voice Signature saved\n\n')
    return embs_dict


def test_recognition(embs_dict, model):
    enroll_embs = np.array(list(embs_dict.values()))
    speakers = list(embs_dict.keys())

    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:  # mention source it will be either Microphone or audio files.
        print("Test phase: Speak anything")
        audio = r.listen(source)  # listen to the source
    text = r.recognize_google(audio, language='fr-FR')

    frames = preprocess_signal(audio.frame_data)

    test_embedding = get_embedding(model, frames, config['maxSeconds'])
    test_embedding = np.array([test_embedding.tolist()])

    print("Comparing to enroll samples....")
    distances = cdist(enroll_embs, test_embedding, metric=config['model']['costMetric'])
    for i in range(len(distances)):
        print('Distance to {} : {}'.format(speakers[i], distances[i][0]))
    ind_min_distance = np.argmin(distances, axis=0)
    print('{} is speaking, and said {}'.format(speakers[ind_min_distance[0]], text))


if __name__ == '__main__':

    print("Loading model weights")
    model = vggvox_model()
    model.load_weights(config['model']['weights'])

    print('Enter your name : ')
    name = input()

    with open(config['enrollmentFile'], 'rb') as f:
        embs_dict = pickle.load(f)

    if name in list(embs_dict.keys()):
        print('You are already enrolled, I will try to recognize you')
    else:
        print('You are not enrolled yet, I will first save your signature and then try to recognize you')
        embs_dict = enrollment(name, embs_dict, model)

    test_recognition(embs_dict, model)
