import time
from threading import Thread
from multiprocessing import Process
from hw1.fibonacci import Fibonacci


def get_multiprocessing_time(func, n, num_processes=10):
    start_multiprocessing_time = time.time()

    processes = []
    for _ in range(num_processes):
        process = Process(target=func, args=(n,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    return time.time() - start_multiprocessing_time


def get_threads_time(func, n, num_threads=10):
    start_threads_time = time.time()

    threads = []
    for _ in range(num_threads):
        thread = Thread(target=func, args=(n,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return time.time() - start_threads_time


if __name__ == '__main__':
    n = 100
    multiprocessing_time = get_multiprocessing_time(Fibonacci(), n)

    threads_time = get_threads_time(Fibonacci(), n)

    with open("artifacts/task_1.txt", "w+") as file:
        file.write(f"Multiprocessing time: {multiprocessing_time}\nThreads time: {threads_time}")
