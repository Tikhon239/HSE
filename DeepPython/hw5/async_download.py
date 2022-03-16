import asyncio
import os

from aiohttp import ClientSession


async def async_download_file(url, session, save_path):
    async with session.get(url) as response:
        r = await response.read()
        with open(save_path, "wb+") as file:
            file.write(r)


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
