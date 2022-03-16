import asyncio
import os

from time import time
from aiohttp import ClientSession
from requests import Session


def sync_download_file(url, session, save_path):
    with session.get(url) as response:
        r = response.content
        with open(save_path, "wb+") as file:
            file.write(r)


async def async_download_file(url, session, save_path):
    async with session.get(url) as response:
        r = await response.read()
        with open(save_path, "wb+") as file:
            file.write(r)


def sync_download_images(url, n_files, save_dir):
    with Session() as session:
        for indx in range(n_files):
            save_path = os.path.join(save_dir, f'sync_{indx}.jpg')
            sync_download_file(url, session, save_path)


def async_download_images(url, n_files, save_dir):
    async def _async_download_images(url, n_files, save_dir):
        async with ClientSession() as session:
            tasks = []
            for indx in range(n_files):
                save_path = os.path.join(save_dir, f'async_{indx}.jpg')
                task = asyncio.create_task(async_download_file(url, session, save_path))
                tasks.append(task)

            return await asyncio.gather(*tasks)

    asyncio.run(_async_download_images(url=url, n_files=n_files, save_dir=save_dir))


if __name__ == "__main__":
    url = 'https://picsum.photos/200/300'
    n_files = 5
    save_dir = os.path.join('artifacts', 'task_1')

    start_time = time()
    sync_download_images(url=url, n_files=n_files, save_dir=save_dir)
    print('Sync download time:', time() - start_time)

    start_time = time()
    async_download_images(url=url, n_files=n_files, save_dir=save_dir)
    print('Async download time:', time() - start_time)
