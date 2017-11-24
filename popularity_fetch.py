
# -*- coding:utf-8 -*-
import urllib.request
import re


class BDTB:


    def __init__(self,baseUrl):
        self.baseURL = baseUrl


    def getPage(self, synset):

        url = self.baseURL + '?wnid=' + synset
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        print(response.read().decode('utf-8'))
        return response

    def getInfo(self,synset):
        page = self.getPage(synset)
        page = self.getPage(1)
        pattern = re.compile('<h1 class="core_title_txt.*?>(.*?)</h1>', re.S)
        result = re.search(pattern, page)
        if result:
            return result.group(1).strip()
        else:
            return None


baseURL = 'http://image-net.org/synset'
bdtb = BDTB(baseURL)
bdtb.getPage('n03457902')