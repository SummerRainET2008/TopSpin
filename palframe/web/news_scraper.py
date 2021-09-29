#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu

import requests
from newspaper import Article
from bs4 import BeautifulSoup

class NewsScraper(object):
  def __init__(self, url, language='zh'):
    self._url = url
    self._article = Article(self._url, language=language)
    self._article.download()
    self._article.parse()
    self._article.nlp()
    self._publish_time = self._get_newstime()

  def _get_newstime(self):
    news_content = requests.get(self._url)
    news_content.encoding = 'utf-8'
    news_content = news_content.text
    soup = BeautifulSoup(news_content, 'html.parser')
    news_time = soup.find(class_="time").text
    #todo: convert the date string to python DateTime object.
    return news_time

  def get_result(self):
    return {
      'date': self._publish_time,
      'title': self._article.title,
      'summary': self._article.summary,
      'url': self._url,
      "text": self._article.text
    }
