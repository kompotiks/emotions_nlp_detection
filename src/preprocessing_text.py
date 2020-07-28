import re


def parsing_output_vosk(data):
    text = []
    start = []
    end = []

    for sample in data:
        if 'result' in sample:
            for line in sample.split('\n'):
                if 'text' in line:
                    words = re.findall(r'[а-яА-ЯёЁ]+', line)
                    for i in words:
                        text.append(i)
                elif 'start' in line:
                    start.append(float(line[16:-5]))
                elif 'end' in line:
                    end.append(float(line[14:-5]))

    data = []
    for i, _ in enumerate(start):
        data.append((text[i], start[i], end[i]))

    return data
