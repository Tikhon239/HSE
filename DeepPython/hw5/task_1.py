import os
from time import time

from async_download import async_download_images
from sync_download import sync_download_images


if __name__ == "__main__":
    url = 'https://picsum.photos/200/300'
    n_files = 5
    save_dir = os.path.join('artifacts', 'task_1')

    async_start_time = time()
    async_download_images(url=url, n_files=n_files, save_dir=save_dir)
    async_finish_time = time() - async_start_time

    sync_start_time = time()
    sync_download_images(url=url, n_files=n_files, save_dir=save_dir)
    sync_finish_time = time() - sync_start_time

    with open(os.path.join(save_dir, 'time.txt',), 'w+') as file:
        file.write(f'Async download time: {async_finish_time}\n')
        file.write(f'Sync download time: {sync_finish_time}')

