from vosk import Model, KaldiRecognizer
import sys
import json
import os


def recognition_text(file_path):
    model = Model("model/model_vosk")

    # Large vocabulary free form recognition
    rec = KaldiRecognizer(model, 16000)

    wf = open(file_path, "rb")
    wf.read(44) # skip header

    while True:
        data = wf.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())

    res = json.loads(rec.FinalResult())
    return res['text']