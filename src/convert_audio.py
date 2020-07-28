import os


def convert_audio(filename, dir_out):
    os.system('ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}'.format(filename, dir_out))