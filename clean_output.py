import glob
from dataclasses import dataclass

@dataclass
class Score:
    para: float
    sts:  float
    sst:  float


def is_score_line(line):
    if 'accuracy:' in line:
        return True
    if 'correlation:' in line:
        return True
    return False


def score_lines(file):
    for line in file:
        if not is_score_line(line):
            continue
        yield line

def parse_score(line):
    dirty_string = line.split(':')[1]
    clean_string = dirty_string.strip()
    return float(clean_string)

def score_values(file_path):
    with open(file_path) as file:
        for line in score_lines(file):
            yield parse_score(line)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def clean_output(file_path: str):
    scores = list(score_values(file_path))
    scores = [Score(a, b, c) for a, b, c, in chunks(scores, 3)]

    for i, (train_score, dev_score) in enumerate(chunks(scores, 2)):
        print(f'Epoch #{i}')
        print(f'train_score: {train_score}')
        print(f'dev_score:   {dev_score}')
        print('-' * 52)

for file_path in glob.glob('slurm_files/*.out'):
    print(file_path)
    print('=' * 52)
    clean_output('output.out')