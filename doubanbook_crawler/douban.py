import requests
import re
import pymysql
import time

# 连接数据库及数据光标，为后面存储数据准备
conn = pymysql.connect(host = '127.0.0.1',user = 'root',passwd = '123456',db = 'pc',port = 3306,charset = 'utf8') # 请输入自己的数据库密码与需要连接的数据库
cursor = conn.cursor()#使用cursor操作数据库以及控制进程

# 定义爬取函数

def get_info(url):
    try:
        res = requests.get(url)
        print(res.text)
        infos = re.findall('<div class="pl2".*?title=(.*?)>.*?<p class="pl">(.*?)/.*?<span class="rating_nums">(.*?)</span>',res.text,re.S)
    except:
        pass

    for info in infos:
        book_name = info[0].strip()
        author = info[1]
        score = info[2]
        cursor.execute("insert into douban_book (book_name,author,score) values (%s,%s,%s)", (book_name,author,str(score))) # 存储数据


# 运行程序
if __name__ == "__main__":
    urls = ['https://book.douban.com/top250?start={}'.format(num) for num in range(0,250,25)]
    print(urls)
    for url in urls:
        get_info(url)
        time.sleep(1)
    conn.commit()
conn.close()