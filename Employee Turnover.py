#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_excel('1673873196_hr_comma_sep.xlsx')


# In[3]:


data.head()


# In[4]:


data.isna().sum()


# In[5]:


# Renaming
data = data.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data.dtypes


# In[9]:


# Looks like about 76% of employees stayed and 24% of employees left. 
# NOTE: When performing cross validation, its important to maintain this turnover ratio
turnover_rate = data.turnover.value_counts() / 14999


# In[10]:


turnover_rate


# In[11]:


data.describe()


# In[12]:


turnover_Summary = data.groupby('turnover')
turnover_Summary.mean()


# In[13]:


#Correlation Matrix
corr = data.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')


# In[14]:


corr


# In[15]:


# Let's compare the means of our employee turnover satisfaction against the employee population satisfaction
emp_population_satisfaction = data['satisfaction'].mean()
emp_turnover_satisfaction = data[data['turnover']==1]['satisfaction'].mean()


# In[16]:


emp_population_satisfaction


# In[17]:


emp_turnover_satisfaction


# In[18]:


import scipy.stats as stats
stats.ttest_1samp(a= data[data['turnover']==1]['satisfaction'], 
                  # Sample of Employee satisfaction who had a Turnover
                  popmean = emp_population_satisfaction)  
                  # Employee Population satisfaction mean


# In[19]:


degree_freedom = len(data[data['turnover']==1])


# In[20]:


LQ = stats.t.ppf(0.025,degree_freedom)  # Left Quartile


# In[21]:


print('left quartile:', LQ)


# In[22]:


# Right Quartile
RQ = stats.t.ppf(0.975,degree_freedom)  
print ('right quartile:', RQ)


# In[27]:


# Graph Employee Satisfaction
sns.distplot(data.satisfaction, kde=False, color="g").set_title('Employee Satisfaction Distribution')
ylabel='Employee Count'


# In[28]:


# Graph Employee Evaluation
sns.distplot(data.evaluation, kde=False, color="r").set_title('Employee Evaluation Distribution')
ylabel='Employee Count'


# In[29]:


# Graph Employee Average Monthly Hours
sns.distplot(data.averageMonthlyHours, kde=False, color="b").set_title('Employee Average Monthly Hours Distribution')
ylabel='Employee Count'


# In[39]:


#salary vs turnover
f, ax = plt.subplots(figsize=(20, 7))
sns.countplot(y="salary", hue='turnover', data=data).set_title('Employee Salary Turnover Distribution');


# In[45]:


#Department V.S. Turnover
ax = plt.subplots(figsize=(20, 7))
sns.countplot(x='department', data=data).set_title('Employee Department Distribution');


# In[46]:


sns.countplot(y="department", hue='turnover', data=data).set_title('Employee Department Turnover Distribution');


# In[47]:


#Turnover vs Projectcount
ax = sns.barplot(x="projectCount", y="projectCount", hue="turnover", data=data, estimator=lambda x: len(x) / len(data) * 100)
ax.set(ylabel="Percent")


# In[48]:


#turnover vs satisfaction
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['turnover'] == 0),'satisfaction'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(data.loc[(data['turnover'] == 1),'satisfaction'] , color='r',shade=True, label='turnover')
plt.title('Employee Satisfaction Distribution - Turnover V.S. No Turnover')


# In[49]:


# Satisfaction VS Evaluation
sns.lmplot(x='satisfaction', y='evaluation', data=data,
           fit_reg=False, # No regression line
           hue='turnover')   


# In[50]:


#K-Means Clustering 
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(data[data.turnover==1][["satisfaction","evaluation"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="satisfaction",y="evaluation", data=data[data.turnover==1],
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Satisfaction")
plt.ylabel("Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Turnover")
plt.show()

Cluster 1 (Blue): Hard-working emp. but not happy with org,
Cluster 2 (Red):  Not Hard-working emp. also not happy with org,
Cluster 3 (Green): Hard-working emp and happy with org
# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler


# In[55]:


# Create dummy variables for the 'department' and 'salary' features, since they are categorical 
department = pd.get_dummies(data=data['department'],drop_first=True,prefix='dep') #drop first column to avoid dummy trap
salary = pd.get_dummies(data=data['salary'],drop_first=True,prefix='sal')
data.drop(['department','salary'],axis=1,inplace=True)
data = pd.concat([data,department,salary],axis=1)


# In[56]:


def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y


# In[57]:


target = 'turnover'
X = data.drop('turnover', axis=1)
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y=data[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=10)


# In[58]:


y_base_rate = base_rate_model(X_test)
from sklearn.metrics import accuracy_score
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))


# In[60]:


# Check accuracy of Logistic Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1)

model.fit(X_train, y_train)
print ("Logistic accuracy is %2.2f" % accuracy_score(y_test, model.predict(X_test)))


# In[62]:


# Using 5 fold Cross-Validation to train our Logistic Regression Model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
kfold = model_selection.KFold(n_splits=5, random_state=7,shuffle=True)
modelCV = LogisticRegression(class_weight = "balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[63]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[64]:


base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print ("Base Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, base_rate_model(X_test)))


# In[65]:


logis = LogisticRegression(class_weight = "balanced")
logis.fit(X_train, y_train)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, logis.predict(X_test)))


# In[66]:


# Decision Tree Model
dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))


# In[67]:


# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    )
rf.fit(X_train, y_train)
print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))


# In[68]:


# Ada Boost
ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
ada.fit(X_train,y_train)
print ("\n\n ---AdaBoost Model---")
ada_roc_auc = roc_auc_score(y_test, ada.predict(X_test))
print ("AdaBoost AUC = %2.2f" % ada_roc_auc)
print(classification_report(y_test, ada.predict(X_test)))


# In[69]:


# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])

plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)

# Plot AdaBoost ROC
plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# In[71]:


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

## plot the importances 
importances = dtree.feature_importances_
feat_names = data.drop(['turnover'],axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

Top 3 Features:
1) Satisfaction
2) YearsAtCompany
3) Evaluation
# In[72]:


import graphviz 
from sklearn import tree
dot_data = tree.export_graphviz(dtree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Employee Turnover") 

dot_data = tree.export_graphviz(dtree, out_file=None, 
                         feature_names=feat_names,  
                         class_names='turnover',  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# In[ ]:




