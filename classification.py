# ../lab-rapidMIner/chapter6/clustering.py
import pandas as pd
import copy as cp
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
from sklearn import metrics
import matplotlib.pyplot as plt


df_2014 = pd.read_csv("./excel/2014_Financial_Data.csv")
df_2015 = pd.read_csv("./excel/2015_Financial_Data.csv")
df_2016 = pd.read_csv("./excel/2016_Financial_Data.csv")
df_2017 = pd.read_csv("./excel/2017_Financial_Data.csv")
df_2018 = pd.read_csv("./excel/2018_Financial_Data.csv")

#delete different attributes
del df_2014['2015 PRICE VAR [%]']
del df_2015['2016 PRICE VAR [%]']
del df_2016['2017 PRICE VAR [%]']
del df_2017['2018 PRICE VAR [%]']
del df_2018['2019 PRICE VAR [%]']

#append data together
df_appended = pd.concat([df_2014, df_2015, df_2016, df_2017],ignore_index = True)
df_appended['StockName'] = df_appended['Unnamed: 0']
del df_appended['Unnamed: 0']

#select attribute
selected_attr = ['StockName','Revenue', 'Market Cap','Revenue per Share', 'Net Income', 'Net Profit Margin','Gross Margin',
                'Gross Profit', 'Income Quality', 'Operating Expenses','Financing Cash Flow', 'Free Cash Flow', 'Free Cash Flow margin','Dividend Yield',
                'PE ratio','Total assets', 'Total current assets','Total liabilities','ROIC', 'EPS Growth','Sector', 'Class']


df_appended = df_appended[selected_attr]

#create dictionary for sector
sector_dict = {'Financial Services': 0 , 'Healthcare': 1, 'Technology':2 , 'Industrials':3 , 'Consumer Cyclical':4, 
                'Basic Materials': 5 , 'Real Estate': 6, 'Energy': 7 , 'Consumer Defensive': 8 , 'Utilities': 9 , 
                'Communication Services': 10}

df_appended['Sector'] = df_appended['Sector'].map(sector_dict)

#drop NaN values
df_appended = df_appended.dropna()


############################# TRAIN ###########################################
#seperate features and labels
features =cp.deepcopy(selected_attr)
features.remove('Class')
features.remove('StockName')
class_names = ['Class']

train_features = df_appended[features]
train_class = df_appended['Class']


# create decision tree
dtree = tree.DecisionTreeClassifier(random_state=1,max_depth=10, min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf= 0.01)
dtree = dtree.fit(train_features, train_class)

# -------------------------------------------------- #
# print text
# print(export_text(dtree, feature_names=features))
# -------------------------------------------------- #

plt.figure(figsize=(100,100))
tree.plot_tree(dtree, fontsize=10, feature_names = features, class_names = ['0','1'] ,filled=True)
plt.savefig('tree.png', dpi = 100)
#plt.show()
#print('Create Decision Tree Completed')


########################### UNSEEN DATA #########################################
df_2018['StockName'] = df_2018['Unnamed: 0']
del df_2018['Unnamed: 0']
df_2018 = df_2018[selected_attr]
df_2018['Sector'] = df_2018['Sector'].map(sector_dict)
df_2018 = df_2018.dropna()

features = cp.deepcopy(selected_attr)
features.remove('Class')
features.remove('StockName')

scoring_features = df_2018[features]

predict_class = dtree.predict(scoring_features)




from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

print("Accuracy:", accuracy_score(df_2018['Class'],predict_class)*100)
print(confusion_matrix(df_2018['Class'], predict_class))
print(classification_report(df_2018['Class'], predict_class))


df_predict_class = pd.DataFrame(dict(Class = predict_class))
df_result = df_2018[selected_attr]
del df_result['Class']
df_result['Class'] = predict_class
reverse_sector_dict = dict((v, k) for k, v in sector_dict.items())
class_toEng = {0:'Not Buy', 1: 'Buy'}
df_result['Sector'] = df_result['Sector'].map(reverse_sector_dict)
df_result['Class'] = df_result['Class'].map(class_toEng)
df_result.to_csv('stock_prediction.csv')


