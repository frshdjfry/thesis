import os
import sys
import urllib.request

from lxml import html
from pip._vendor import requests

BASE_URL = 'http://www1.cpdl.org'


def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_links(content):
    tree = html.fromstring(content)
    container = tree.cssselect("div.mw-parser-output")[2]
    links = container.cssselect("a")
    res = []
    for l in links:
        res.append(l.attrib.get('href'))
    return res


def get_html(input_file):
    with open(input_file) as f:
        return f.read()


def download_content(l):
    response = requests.get(l)
    return response.content


def get_mxl_link(html_content):
    tree = html.fromstring(html_content)
    download_links = tree.cssselect('a.internal')
    res = []
    for download_link in download_links:
        if '.mxl' in download_link.attrib.get('href'):
            res.append(BASE_URL + download_link.attrib.get('href'))
    return res[0] if res else []


def save_mxl(mxl_link, out_dir):
    create_folder_if_not_exists(out_dir)
    testfile = urllib.request.urlretrieve(mxl_link, os.path.join(os.getcwd(), out_dir, mxl_link.split('/')[-1]))
    print(testfile)


def main(input_file, out_dir):
    html_content = get_html(input_file)
    links = get_links(html_content)
    counter = 0
    begin = False
    for l in links:
        if l == 'http://www1.cpdl.org/wiki/index.php/Madrigal,_Op._35_(Gabriel_Faur%C3%A9)':
            begin = True
        if begin:
            try:
                html_content = download_content(l)
                mxl_link = get_mxl_link(html_content)
                save_mxl(mxl_link, out_dir)
                counter += 1
            except Exception as e:
                print(e)
                print('failed to load ', l)
    print('%s file from %s article downloaded' % (counter, len(links)))


if __name__ == '__main__':
    print('params: input file, output directory')
    main(sys.argv[1], sys.argv[2])
