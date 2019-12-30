import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
db=pd.read_csv("t250.csv",encoding='utf-8')

db=db.dropna()
ds=db.sort_values('score',ascending=False)[: 10]
plt.style.use('ggplot')

plt.rcParams['font.family'] = ['SimHei']

fig, ax = plt.subplots(figsize=(10,8))

y_pos = np.arange(0,20, 2)

score = ds['score']
ax.barh( y_pos, score, align='center', height=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(ds['book_name'])
ax.invert_yaxis()  
ax.set_xlabel('评分')
ax.set_title('评分top10图书')

plt.show()
fig, ax = plt.subplots()
n, bins, patches = ax.hist(db['score'])
ax.plot((bins + 0.1/2)[:-1], n, 'w--')
ax.set_xlabel('评分')
ax.set_ylabel('评分数量')
ax.set_title('评分分布')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

colors= plt.get_cmap('gist_ncar')(np.linspace(0.15, 0.85, 35))
fig, ax = plt.subplots(figsize=(10,8))
y = np.arange(0,20,2)
ylabels = db.groupby('author')['score'].mean().dropna()[1:].sort_values(ascending=False)[:10].index
rate = db.groupby('author')['score'].mean().dropna()[1:].sort_values(ascending=False)[:10].values
ax.barh(y,rate, align='center', height=1.2, color=colors)
ax.set_yticks(y)
ax.set_yticklabels(ylabels)
ax.set_xlabel('平均评分')

ax.set_title('作者平均评分')
ax.invert_yaxis()
plt.tight_layout()
plt.show()


