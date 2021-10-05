import numpy as np
from scipy.stats import rankdata
from argparse import ArgumentParser


def read_data(file_name: str):
    x_array = []
    y_array = []
    with open(file_name) as file:
        for line in file:
            x, y = map(int, line.split())
            x_array.append(x)
            y_array.append(y)
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    indexes = np.argsort(x_array)
    return x_array[indexes], y_array[indexes]


def monotonic_conjugation(input_file_name: str, output_file_name: str):
    _, y_array = read_data(input_file_name)

    n = len(y_array)
    if n <= 9:
        raise ValueError('n must be greater than 9')

    ranks = rankdata(y_array)
    ranks = -(ranks - ranks.max() - 1)

    p = round(n / 3)
    r1 = ranks[:p].sum()
    r2 = ranks[-p:].sum()

    diff = round(r1 - r2)
    error = round((n + 0.5) * np.sqrt(p / 6))
    conjugation =  round((r1 - r2) / (p * (n - p)), 2)

    with open(output_file_name, 'w') as file:
        file.write(f'{diff} {error} {conjugation}\n')


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--input', type=str, default='in.txt')
    args.add_argument('--output', type=str, default='out.txt')

    args = args.parse_args()
    monotonic_conjugation(args.input, args.output)