import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
sns.set()
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_circles
import pandas as pd
import seaborn as sns
import mlxtend
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier        
from sklearn import metrics   


data=pd.read_csv('latest.csv', header=0)


py=list(data['publishyear'])
for i in range(len(py)):
    py[i]=2019-py[i]
data['publishyear']=py



data['score']=data['score']*10
data['score']


cols = ['score','num', 'publishyear', 'rank']
sns.pairplot(data[cols], height=4)
plt.tight_layout()
plt.show()
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=3)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.3f', annot_kws={
                 'size': 20}, yticklabels=cols, xticklabels=cols)
plt.show()



from sklearn.decomposition import PCA


pca=PCA(n_components=3)
X=data[cols]
NX=pca.fit_transform(X)
NX=abs(NX)


import seaborn as sns; sns.set()  # for plot styling



plt.figure(figsize=(20, 20), dpi=80)
plt.scatter(NX[:, 0], NX[:, 1],c=NX[:, 2],cmap=plt.cm.autumn);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(NX)
y_kmeans = kmeans.predict(NX)

plt.figure(figsize=(20, 20), dpi=80)

plt.scatter(NX[:, 0], NX[:, 1], c=y_kmeans, s=10, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5);
plt.figure(figsize=(30, 30), dpi=80)

plt.scatter(NX[:, 0], NX[:, 1], c=y_kmeans, s=10, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5);

from sklearn.cluster import DBSCAN

y_pred = DBSCAN(eps = 10000, min_samples = 3).fit_predict(NX)

plt.figure(figsize=(20, 20), dpi=80)
plt.scatter(NX[:, 0], NX[:, 1], c=y_pred,cmap='viridis')

plt.show()

plt.figure(figsize=(50, 50), dpi=80)
plt.scatter(NX[:, 0], NX[:, 1], c=y_pred,cmap='viridis')

plt.show()


X1=NX[NX[:,0]<25000] 
rk=X1[:,0]
cm=X1[:,1]

plt.figure(figsize=(20, 50), dpi=80)
plt.scatter(rk, cm, cmap=plt.cm.autumn)

plt.show()
#rk=rk[:,0]
#rk=np.array[rk]
def auto_plot(df_obj, title=None):
    plt.plot(df_obj.x1[df_obj.target == 1], df_obj.x2[df_obj.target == 1], 'b^', label='class 1')
    plt.plot(df_obj.x1[df_obj.target == 0], df_obj.x2[df_obj.target == 0], 'rs', label='class 0')
    plt.title(title)
    plt.xlabel('rank')
    plt.ylabel('comments')
    plt.legend();

cm=np.array(cm)
mydict={'x1':rk, 'x2':cm}
df = pd.DataFrame(mydict)
df['target'] = 0
for i,_ in df.iterrows():
    if df.loc[i, 'x1'] <= 10000:
        df.loc[i, 'target'] = 1


df2=df[0:130]

X_train = df2[['x1', 'x2']]
y_train = df2.target

df4=df[130:]

X_test = df4[['x1', 'x2']]
y_test = df4.target

auto_plot(df2, title='Separable')


from sklearn.neighbors import KNeighborsClassifier

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
   
  
    ax = ax or plt.gca()
    ax.set_xlabel('comments')
    ax.set_ylabel('rank')
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('on')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
  


lsvc = LinearSVC(penalty='l2', loss='hinge', random_state=42,C=2)
lsvc.fit(X_train, y_train)


plot_decision_regions(X_train.values, y_train.values, clf=lsvc,res=0.001, legend=2)
plt.title('Decision Regions')
plt.xlabel('rank')
plt.ylabel('comments')

lsvc.predict(X_test)

print('score: {}'.format(lsvc.score(X_test, y_test)))


X_test1=np.array(X_test)
y_test1=np.array(y_test)
X_train1=np.array(X_train)
y_train1=np.array(y_train)
visualize_classifier(LR, X_train1, y_train1)


LR = linear_model.LogisticRegression(solver='lbfgs', max_iter = 1000,multi_class='auto')
LR.fit(X_train, y_train)
LR.predict(X_test)
#print(y_test)
print('score: {}'.format(LR.score(X_test１, y_test１)))


tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
# Grow a Decision Tree
tree_clf.fit(X_test, y_test)
from sklearn.tree import export_graphviz
export_graphviz(tree_clf, out_file='tree.dot',
                feature_names=['x1', 'x2'],
                class_names=['Yellow', 'Blue', 'Red'],
                rounded=True, filled=True)
visualize_classifier(DecisionTreeClassifier(), X_test１, y_test１)
visualize_classifier(DecisionTreeClassifier(), X_train1, y_train1)





rf=RandomForestClassifier(n_estimators=10000,criterion = 'entropy', random_state = 0)





rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))



from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(cm)




print(cr)



visualize_classifier(rf, X_train1, y_train1)



from sklearn.tree import export_graphviz

export_graphviz(tree_clf, out_file='tree.dot',
                feature_names=['x1', 'x2'],
                class_names=['Yellow', 'Blue', 'Red'],
                rounded=True, filled=True)





