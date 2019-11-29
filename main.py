import numpy as np
from scipy.spatial.distance import cdist
from model import vggvox_model
from utils import remove_dc_and_dither, normalize_frames, build_buckets, read_config
import sigproc
import speech_recognition as sr
import pickle

config = read_config('./resources/config.json')


def get_embedding(model, fft_signal):
    return np.squeeze(model.predict(fft_signal.reshape(1, *fft_signal.shape, 1)))


def preprocess_signal(audio, max_sec):
    signal = np.fromstring(audio, dtype=np.int16)
    signal = signal / 32768
    signal *= 2 ** 15
    signal = remove_dc_and_dither(signal, config['sampleRate'])
    signal = sigproc.preemphasis(signal, coeff=config['preemphasisAlpha'])
    frames = sigproc.framesig(signal, frame_len=config['frameLength'] * config['sampleRate'], frame_step=config['frameStep'] * config['sampleRate'], winfunc=np.hamming)

    buckets = build_buckets(max_sec, config['bucketStep'], config['frameStep'])
    fft = abs(np.fft.fft(frames, n=config['numFFT']))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1] - rsize) / 2)
    fft_signal = fft_norm[:, rstart:rstart + rsize]
    return fft_signal


def enrollment(name, embs_dict, model):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:  # mention source it will be either Microphone or audio files.
        print("Enrollment phase, say :  \"J'aimerais m'enregister et j'accepte que ma signature vocale soit sauvegardée\"")
        audio = r.listen(source)  # listen to the source

    fft_signal = preprocess_signal(audio.frame_data, config['maxSeconds'])

    new_embedding = get_embedding(model, fft_signal)
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

    frames = preprocess_signal(audio.frame_data, config['maxSeconds'])

    test_embedding = get_embedding(model, frames)
    test_embedding = np.array([test_embedding.tolist()])

    print("Comparing to enroll samples....")
    distances = cdist(enroll_embs, test_embedding, metric=config['model']['costMetric'])
    for i in range(len(distances)):
        print('Distance to {} : {}'.format(speakers[i], distances[i][0]))
    ind_min_distance = np.argmin(distances, axis=0)
    if distances[ind_min_distance] < config['signatureThreshold']:
        return speakers[ind_min_distance[0]], True, text
    else:
        print('You are not enrolled yet, please enter your name and I will save your signature :')
        name = input()
        return name, False, text


if __name__ == '__main__':

    print("Loading model weights")
    model = vggvox_model()
    model.load_weights(config['model']['weights'])

    with open(config['enrollmentFile'], 'rb') as f:
        embs_dict = pickle.load(f)

    # del embs_dict['pedro']
    # with open(config['enrollmentFile'], 'wb') as f:
    #     pickle.dump(embs_dict, f, pickle.HIGHEST_PROTOCOL)

    name, enrolled, text = test_recognition(embs_dict, model)
    if enrolled:
        print('\n{} is speaking and said : \"{}\"'.format(name, text))
    else:
        enrollment(name, embs_dict, model)