# -*- coding: utf-8 -*-
import os
import csv
import re
import requests 
import io
import sys
import time
import random
topnum = 1



def getHtml(url):
    try:
        page = requests.get(url,headers=kv)
        page.raise_for_status()
        html = page.text
    except:
        print("failed to geturl")
        return ''
    else:
        return html


def getTitle(html):
    nameList = re.findall(r'<a href="https.*?".*?target="_blank">(.*?)</a>',html,re.S)
    newNameList = [];
    global topnum
    for index,item in enumerate(nameList):
        if item.find("img") == -1:
         
            if topnum%26 !=0:
                newNameList.append(item);
            topnum += 1;
    return newNameList


def getDetail(html):
    detailList = re.findall(r'<a href="(https.*?)".*?target="_blank">.*?</a>',html,re.S)
    newDetailList = []
    for index,item in enumerate(detailList):
        if item.find("subject") != -1 and index % 2!=0:
            newDetailList.append(item);
         

    return newDetailList


def getPublishYear(html):
    publishYearList = re.findall(r'<span class="pl">出版年.*?</span>(.*?)<br/>',html,re.S)
    return publishYearList


def getPress(html):
    pressList = re.findall(r'<span class="pl">出版社.*?</span>(.*?)<br/>',html,re.S)
    return pressList


def getIsbn(html):
    isbnList = re.findall(r'<span class="pl">ISBN.*?</span>(.*?)<br/>',html,re.S)
    return isbnList


def getImg(html):
    imgList = re.findall(r'img.*?width=.*?src="(http.*?)"',html,re.S)
    newImgList = []
    for index,item in enumerate(imgList):
        if item.find("js") == -1 and item.find("css") == -1 and item.find("dale") == -1 and item.find("icon") == -1 and item.find("png") == -1:
            newImgList.append(item);

    return newImgList;
def getScore(html):
    scoreList = re.findall(r'<span.*?class="rating_nums">(.*?)</span>',html,re.S)
    return scoreList

def getComment(html):
    commentList = re.findall(r'<span>(.*?)</span>',html,re.S)
    newcommentList =[]
    for index,item in enumerate(commentList):
        if item.find("评价") >= 1:
            newcommentList.append(item);

    return newcommentList

def saveInfo(infoList):
    with open('/home/mark/桌面/321.csv','w+',newline='',encoding='gb18030') as fp:
        a = csv.writer(fp,delimiter = ',')
        a.writerow(['书  名','评  分','评价人数','图片链接','出 版社','出版年份',' ISBN '])
        a.writerows(infoList)
        print('保存完毕')

namesUrl = []
imgsUrl = []
scoresUrl = []
commentsUrl = []
detailsUrl = []
introductionsUrl = []
isbnsUrl = []
publishYearsUrl = []
newPresssUrl = []
allInfo = []
kv={'user-urgent':'Mozilla/5.0'}
print ("Starting Main \n 普通爬取开始时时间")
print(time.ctime(time.time()))


for page in range(0,450,25):
    url = "https://www.douban.com/doulist/1264675/?start={}".format(page)
    html = getHtml(url);
    if html == '':
        namesUrl.extend('none');
        imgsUrl.extend('none')
        scoresUrl.extend('none')
        commentsUrl.extend('none')
        introductionsUrl.extend('none')
    else:
        namesUrl.extend(getTitle(html))
        imgsUrl.extend(getImg(html))
        scoresUrl.extend(getScore(html))
        commentsUrl.extend(getComment(html))
        introductionsUrl.extend(getDetail(html))


namesUrl.pop()


for index,item in enumerate(introductionsUrl):
    print(item)
    if getHtml(item) == '':
        newPresssUrl.append("该链接不存在")
        publishYearsUrl.append("该链接不存在")
        isbnsUrl.append("该链接不存在")
    else:
        html_detail=getHtml(item)
       
        newPresssUrl.append(getPress(html_detail))
        publishYearsUrl.append(getPublishYear(html_detail))
        isbnsUrl.append(getIsbn(html_detail))
        time.sleep(random.randint(1,2))

for i in range(0,len(namesUrl)):
    tmp=[]
    tmp.append(namesUrl[i])
    tmp.append(scoresUrl[i])
    tmp.append(commentsUrl[i])
    tmp.append(imgsUrl[i])
    tmp.append(newPresssUrl[i])
    tmp.append(publishYearsUrl[i])
    tmp.append(isbnsUrl[i])

    allInfo.append(tmp)

print(len(namesUrl))
print(len(commentsUrl))
print(len(imgsUrl))
print(len(scoresUrl))
print(len(newPresssUrl))
print(len(publishYearsUrl))
print(len(isbnsUrl))

saveInfo(allInfo)
print ("Exiting Main \n 普通爬取结束时时间")
print(time.ctime(time.time()))
