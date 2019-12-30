import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
db=pd.read_csv("9p.csv",encoding='utf-8')
db=db.dropna()
db.publishyear=pd.to_numeric(db.publishyear,downcast='integer')
l=db[db['publishyear']>2010]
l1=l.sort_values('score',ascending=False)[: 20]
plt.style.use('ggplot')

plt.rcParams['font.family'] = ['SimHei']

fig, ax = plt.subplots(figsize=(10,8))

y_pos = np.arange(0,40, 2)

score = l1['score']
ax.barh( y_pos, score, align='center', height=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(l1['book_name'])
ax.invert_yaxis() 
ax.set_xlabel('评分')
ax.set_title('2000年以后出版的评分top20图书')

plt.show()

l2=l.sort_values('num',ascending=False)[:20]
plt.style.use('ggplot')

plt.rcParams['font.family'] = ['SimHei']

fig, ax = plt.subplots(figsize=(10,8))

y_pos = np.arange(0,40, 2)

comments = l2['num']
ax.barh( y_pos, comments, align='center', height=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(l2['book_name'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('评论')
ax.set_title('2000年以后出版的评论top20图书')

plt.show()
d1=db.sort_values('num',ascending=False)[:10]
plt.style.use('ggplot')

plt.rcParams['font.family'] = ['SimHei']

fig, ax = plt.subplots(figsize=(10,8))

y_pos = np.arange(0,20, 2)

comments = d1['num']
ax.barh( y_pos, comments, align='center', height=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(d1['book_name'])
ax.invert_yaxis()  
ax.set_xlabel('评论')
ax.set_title('评论top10图书')

plt.show()

colors= plt.get_cmap('gist_ncar')(np.linspace(0.15, 0.85, 35))
fig, ax = plt.subplots(figsize=(10,8))
y = np.arange(0,76,2)
ylabels = db.groupby('publishyear')['num'].mean().dropna()[1:].sort_values(ascending=False).index
rate = db.groupby('publishyear')['num'].mean().dropna()[1:].sort_values(ascending=False).values
ax.barh(y,rate, align='center', height=1.2, color=colors)
ax.set_yticks(y)
ax.set_yticklabels(ylabels)
ax.set_xlabel('平均评论指数')

ax.set_title('各年份出版书籍平均评论指数')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
n, bins, patches = ax.hist(db['score'])
ax.plot((bins + 0.1/2)[:-1], n, 'w--')
ax.set_xlabel('评分')
ax.set_ylabel('评分数量')
ax.set_title('评分分布')

fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
data = db['num'][ db['num']!=0]
ax.set_xticks([1])
ax.set_xticklabels(['comments'])
ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=True)
ax.vlines(1,25, db['num'][ db['num']!=0].describe()['75%'], color='k', linestyle='-', lw=20)
ax.scatter(1, db['num'][ db['num']!=0].describe()['50%'], marker='o', color='white', s=30, zorder=5)
ax.set_title('评论分布')
plt.show()



