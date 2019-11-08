import pymysql
conn = pymysql.connect(host = '127.0.0.1',user = 'root',passwd = '123456',db = 'sys',port = 3306,charset = 'utf8') # 请输入自己的数据库密码与需要连接的数据库
cursor = conn.cursor()
sql = "create table douban9f(book_name text,score text,num text,picturelink text,publisher text,pulishyear text,ISBN text)engine innodb default charset=utf8"
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 向数据库提交
   conn.commit()
except:
   # 发生错误时回滚
   conn.rollback()
conn.close()