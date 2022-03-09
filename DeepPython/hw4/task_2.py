import os
import logging
from logging import FileHandler
from logging import Formatter
import math
import time
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial


def integrate(f, a, b, *, n_jobs=1, n_iter=1000):
    acc = 0
    step = (b - a) / n_iter
    for i in range(n_iter):
        acc += f(a + i * step) * step
    return acc


def threads_integrate(f, a, b, *, n_jobs=1, n_iter=1_000):
    start_threads_time = time.time()
    acc = 0
    locker = threading.Lock()
    step = (b - a) / n_iter
    points = [a + i * step for i in range(n_iter)]

    def integrate_step(point):
        nonlocal f, step, acc, locker
        with locker:
            threads_logger.info(f"Start calc f(x): {str({'x': round(point, 4), 'step': step})}")

        f_x = f(point)
        d_x = f_x * step

        with locker:
            threads_logger.info(f"f(x) calculated, atomic increase acc: {str({'f(x)': round(f_x, 4), 'cur acc': round(acc, 4)})}")
            acc += d_x
        return d_x

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        executor.map(integrate_step, points)

    return time.time() - start_threads_time, acc


def multiprocess_integrate_step(f, step, locker, point):
    with locker:
        multiprocess_logger.info(f"Start calc f(x): {str({'x': round(point, 4), 'step': step})}")

    f_x = f(point)
    d_x = f_x * step

    with locker:
        multiprocess_logger.info(f"f(x) calculated, atomic increase acc: {str({'f(x)': round(f_x, 4), 'cur acc': round(acc.value, 4)})}")
        with acc.get_lock():
            acc.value += d_x
    return d_x


def multiprocess_integrate(f, a, b, *, n_jobs=1, n_iter=1_000):
    start_threads_time = time.time()
    local_acc = 0
    locker = multiprocessing.Manager().Lock()
    step = (b - a) / n_iter
    points = [a + i * step for i in range(n_iter)]

    func = partial(multiprocess_integrate_step, f, step, locker)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(func, point) for point in points]
        for future in as_completed(futures):
            local_acc += future.result()

    return time.time() - start_threads_time, local_acc, acc.value


acc = multiprocessing.Value('f', 0)

multiprocess_logger = logging.getLogger('multiprocess.logger')
multiprocess_logger.setLevel(logging.INFO)
file_handler_1 = FileHandler(os.path.join('artifacts', 'multiprocess_integrate_log.txt'))
file_handler_1.setLevel(logging.INFO)
file_handler_1.setFormatter(Formatter("%(asctime)s; %(message)s"))
multiprocess_logger.addHandler(file_handler_1)

threads_logger = logging.getLogger('threads.logger')
threads_logger.setLevel(logging.INFO)
file_handler_2 = FileHandler(os.path.join('artifacts', 'threads_integrate_log.txt'))
file_handler_2.setLevel(logging.INFO)
file_handler_2.setFormatter(Formatter("%(asctime)s; %(message)s"))
threads_logger.addHandler(file_handler_2)

if __name__ == "__main__":
    cpu_num = os.cpu_count()

    with open("artifacts/task_2.txt", "w+") as file:
        file.write('n_jobs\tprocess\tthread\n')
        for n_jobs in range(1, 2 * cpu_num + 1):
            open(os.path.join('artifacts', 'multiprocess_integrate_log.txt'), 'w').close()
            open(os.path.join('artifacts', 'threads_integrate_log.txt'), 'w').close()

            multiprocessing_time = multiprocess_integrate(math.cos, 0, math.pi / 2, n_jobs=n_jobs, n_iter=10_000)[0]
            threads_time = threads_integrate(math.cos, 0, math.pi / 2, n_jobs=n_jobs, n_iter=10_000)[0]

            file.write(f'{n_jobs}\t{multiprocessing_time}\t{threads_time}\n')
