import asyncio
import os

import aiofiles
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup


async def get_html(url, session):
    try:
        response = await session.request(method="GET", url=url)
        response.raise_for_status()
        html = await response.text()
    except (
            aiohttp.ClientError,
            aiohttp.http_exceptions.HttpProcessingError,
    ) as e:
        print(getattr(e, "message", "no message"))
        return None
    else:
        return html


async def get_soup(url, session):
    html = await get_html(url, session)
    if html is None:
        return None
    return BeautifulSoup(html, "html.parser")


async def async_add_url(url, session, urls, urls_save_path):
    soup = await get_soup(url, session)
    if soup is None:
        return None

    results = soup.find_all('a', class_='_93444fe79c--link--eoxce', href=True)
    results = map(lambda x: x['href'], results)

    async with aiofiles.open(urls_save_path, "a") as f:
        for result in results:
            if result not in urls:
                await f.write(f"{result}\n")


async def async_add_data(url, session, data_save_path):
    soup = await get_soup(url, session)
    if soup is None:
        return None

    results = soup.find_all(class_="a10a3f92e9--title--UEAG3")
    results = map(lambda x: x.get_text(), results)

    async with aiofiles.open(data_save_path, "a") as f:
        for result in results:
            await f.write(f"{url}\n{result}\n")


async def async_update_urls(urls, urls_save_path, n_pages=1):
    async with ClientSession(headers={"Referer": "https://www.cian.ru"}) as session:
        url = 'https://www.cian.ru/cat.php?deal_type=sale&offer_type=flat&p=3&region=1'
        tasks = []
        for page_id in range(1, n_pages + 1):
            cur_url = f'{url}&engine_version={page_id}'
            task = asyncio.create_task(async_add_url(cur_url, session, urls, urls_save_path))
            tasks.append(task)
        return await asyncio.gather(*tasks)


async def async_update_data(urls, data_save_path):
    async with ClientSession(headers={"Referer": "https://www.cian.ru"}) as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(async_add_data(url, session, data_save_path))
            tasks.append(task)
        return await asyncio.gather(*tasks)


def get_urls(urls_save_path):
    urls = set()
    with open(urls_save_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            urls.add(line.strip())
    return urls


def async_download_pages(urls_save_path, data_save_path, n_pages=1):
    urls = get_urls(urls_save_path)
    asyncio.run(async_update_urls(urls=urls, urls_save_path=urls_save_path, n_pages=n_pages))
    new_urls = get_urls(urls_save_path) - urls
    asyncio.run(async_update_data(urls=new_urls, data_save_path=data_save_path))


if __name__ == "__main__":
    n_pages = 3
    urls_save_path = os.path.join('artifacts', 'task_2', 'urls.txt')
    data_save_path = os.path.join('artifacts', 'task_2', 'data.txt')

    async_download_pages(urls_save_path=urls_save_path, data_save_path=data_save_path, n_pages=n_pages)
