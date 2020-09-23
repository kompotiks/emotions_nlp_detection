from src.split_audio import split_audio
from src.convert_audio import convert_audio
from src.predict_nlp import prediction
from src.recognition import recognition_text

import operator
import os
import glob
import argparse


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?')
    return parser


def remove_time_file(path: str):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)


def count_result(data: str, emo: list) -> list:
    values_emotions = {}
    data_str = [d[0] for d in data]

    for e in emo:
        values_emotions.update({e: data_str.count(e)})
    return sorted(values_emotions.items(), key=operator.itemgetter(1), reverse=True)[:2]


def write_txt(result: list):
    with open('text_out.txt', 'w') as f:
        f.write(f'{result[0]}{result[1]}')


def use_module(path=None) -> list:
    parser = createParser()
    namespace = parser.parse_args()

    if not path: path = str(namespace.path)

    chunks_dir = 'data/chunks/'
    chunks_dir_text = 'data/chunks_text/'
    emo = ['angry', 'fear', 'happy', 'love', 'sadness', 'surprise']

    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
    else:
        remove_time_file(chunks_dir)

    if not os.path.exists(chunks_dir_text):
        os.makedirs(chunks_dir_text)
    else:
        remove_time_file(chunks_dir_text)

    split_audio(path)
    for file in os.listdir(chunks_dir):
        convert_audio(chunks_dir + file, chunks_dir_text + file)

    raw_text = []
    for file in os.listdir(chunks_dir_text):
        raw_text.append(recognition_text(chunks_dir_text + file))

    result_model_full = []
    result_model = []
    for sample in raw_text:
        # sample = ' '.join([text for text in sample])
        result = prediction(sample).cpu().detach().numpy()
        result_model_full.append(result)
        arg = result.argmax()
        result_model.append((emo[arg], (result[0][arg])))

    result = count_result(result_model, emo)
    write_txt(result)

    return result


if __name__ == '__main__': print(use_module())
