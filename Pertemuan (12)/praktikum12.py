from bs4 import BeautifulSoup as Soup
import os
import re
import requests

def downloader(link):
    req = requests.get(link)
    req.encoding = "utf8"
    return req.text

contents = downloader("https://jurnal.stis.ac.id/index.php/jurnalasks/")
# print(contents)

soup = Soup(contents, "lxml")
# print(soup.prettify())

# print(soup.title)

# print(soup.find_all("a", attrs={"id": "article-532"}))

urls = soup.find_all("a", attrs={"id": re.compile(r"(article)")})

for u in urls:
    content_u = downloader(u['href'])
    soup_u = Soup(content_u, "lxml")
    print(soup_u.find("h1", attrs={"class": "page_title"}).text)