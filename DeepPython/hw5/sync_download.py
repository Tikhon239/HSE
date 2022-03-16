import os

from requests import Session


def sync_download_file(url, session, save_path):
    with session.get(url) as response:
        r = response.content
        with open(save_path, "wb+") as file:
            file.write(r)


def sync_download_images(url, n_files, save_dir):
    with Session() as session:
        for indx in range(n_files):
            save_path = os.path.join(save_dir, f'sync_{indx}.jpg')
            sync_download_file(url, session, save_path)
