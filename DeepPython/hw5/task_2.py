import asyncio
import os

import aiofiles
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup


async def get_html(url, session):
    response = await session.request(method="GET", url=url)
    response.raise_for_status() # raise if status >= 400
    html = await response.text()

    return html


async def parse_page(url, session):
    try:
        html = await get_html(url, session)
    except (
            aiohttp.ClientError,
            aiohttp.http_exceptions.HttpProcessingError,
    ) as e:
        print(getattr(e, "message", "no message"))
        return None
    else:
        soup = BeautifulSoup(html, "html.parser")

        results = soup.find_all(class_="OffersSerpItem__title")
        return map(lambda x: x.get_text(), results)


async def async_download_page(url, session, save_path):
    results = await parse_page(url, session)
    if results is None:
        return None

    async with aiofiles.open(save_path, "a") as f:
        for result in results:
            await f.write(f"{result}\n")


def async_download_pages(url, save_path, n_pages=1):
    async def _async_download_pages(url, save_path, n_pages=1):
        async with ClientSession() as session:
            tasks = []
            for page_id in range(1, n_pages + 1):
                cur_url = f'{url}&page={page_id}'
                task = asyncio.create_task(async_download_page(cur_url, session, save_path))
                tasks.append(task)
            return await asyncio.gather(*tasks)

    asyncio.run(_async_download_pages(url=url, save_path=save_path, n_pages=n_pages))


if __name__ == "__main__":
    url = 'https://realty.yandex.ru/sankt-peterburg/kupit/kvartira/?from=main_menu'
    n_pages = 1
    save_path = os.path.join('artifacts', 'task_2', 'data.txt')

    async_download_pages(url=url, save_path=save_path, n_pages=n_pages)
