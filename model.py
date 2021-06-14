#!/usr/bin/env python
# coding: utf-8

# # Problem statement: 
# 
# The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.
# 
#  
# 
# Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
# 
#  
# 
# With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.
# 
# <ol> to build a sentiment-based product recommendation system, which includes the following tasks.
# 
# <li>Data sourcing and sentiment analysis
# <li>Building a recommendation system
# <li>Improving the recommendations using the sentiment analysis model
# <li>Deploying the end-to-end project with a user interface
#     
#   </ol>

# # Sentiment based recommendation system.
# 
# <ol> <b>  We have to divide the assignment into 3 parts.</b>
#  
# <li> Reading data and data cleaning.
# 
# <li><ol><b> Model building</b>
#     <li>sentiment analysis model
#     <li>Collaberative filtering based recomendation system
#         </ol>
# <li> integration of ML models
# 
# <li> Deployment of models
# </ol>

# ##### Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import *
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.dummy import DummyClassifier
from string import punctuation
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
from nltk import ngrams
from itertools import chain
from fractions import Fraction
import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import WhitespaceTokenizer


# ###### reading data from csv and analysing the source data

# In[2]:


df = pd.read_csv('sample30.csv', sep = ',')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df2 =df.copy()


# In[8]:


df2 =df2[df2['user_sentiment'].isnull()== False]
df2.reset_index(drop=True)


# In[9]:


# Concatenating review title and review text which would be used for sentiment analysis
df2['total_review'] = df2['reviews_title']+', '+ df2['reviews_text']


# In[10]:


df2.info()


# ###### using WordCloud library to visualize the import words in the reviews

# In[11]:


pip install wordcloud


# In[12]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[13]:


show_wordcloud(df2['reviews_text'])


# In[14]:


show_wordcloud(df2['reviews_title'])


# In[15]:


df2['user_sentiment'] = df2['user_sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)


# In[16]:


y= df2.pop('user_sentiment')
X = df2['total_review']


# In[17]:


df2['total_review'].apply(str)


# In[18]:


y.value_counts()


# In[19]:


# visualize the target variable
g = sns.countplot(y)
g.set_xticklabels(['negative','Positive'])
plt.show()


# In[20]:


# Function that returns the wordnet object value corresponding to the POS tag
from nltk.corpus import stopwords
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# Function for cleaning the text

def clean_text(text):
    # lower text
    if type(text) != str:
        text = str(text)
    else:
        text = text

    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    
    # join all
    text = " ".join(text)
    return(text)


# In[21]:


# clean text data
df2["reviews_clean_text"] = df2.apply(lambda x: clean_text(x["total_review"]),axis=1)


# In[22]:


from nltk import word_tokenize


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer

### Creating CountVectorizer
tfidf = TfidfVectorizer(tokenizer= word_tokenize, 
                               stop_words=stopwords.words('english'), 
                               ngram_range=(1,1)) 

tfidf_vec = tfidf.fit_transform(df2["reviews_clean_text"])


# In[24]:


# Saving the vectorizer so that it can be used later while deploying the model

import pickle

# Save to file in the current working directory
pkl_filename = "Tfidf_vectorizer.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tfidf, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickled_tfidf_vectorizer = pickle.load(file)


# In[25]:



# Splitting the data into train and test

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_vec,
                                                                            y,
                                                                            test_size = 0.3,
                                                                            random_state = 100)


# In[26]:


lr = LogisticRegression()
  
lr.fit(X_train_tfidf, y_train_tfidf.ravel())
  
lr_predictions = lr.predict(X_test_tfidf)


# In[27]:


# Confusion matrix 
confusion = confusion_matrix(y_test_tfidf, lr_predictions)
print(confusion)


# In[28]:


# print classification report
print(classification_report(y_test_tfidf, lr_predictions))
print("Accuracy : ",accuracy_score(y_test_tfidf, lr_predictions))
print("F1 score: ",f1_score(y_test_tfidf, lr_predictions))
print("Recall: ",recall_score(y_test_tfidf, lr_predictions))
print("Precision: ",precision_score(y_test_tfidf, lr_predictions))


# In[29]:


print("Imbalance Before OverSampling, counts of label '1': {}".format(sum(y_train_tfidf == 1)))
print("Imbalance Before OverSampling, counts of label '0': {} \n".format(sum(y_train_tfidf == 0)))


# In[30]:


# import SMOTE module from imblearn library
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2)
X_train_sm, y_train_sm = sm.fit_sample(X_train_tfidf, y_train_tfidf.ravel())


# In[31]:


print('After OverSampling, the shape of train_X: {}'.format(X_train_sm.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_sm.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_sm == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_sm == 0)))


# In[32]:



# Training after Smote

lr_sm = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_sm.fit(X_train_sm, y_train_sm.ravel())
predictions_sm = lr_sm.predict(X_test_tfidf)

# Confusion matrix 
confusion_lr_sm = confusion_matrix(y_test_tfidf, predictions_sm)
print(confusion_lr_sm)


# In[33]:


# print classification report

print(classification_report(y_test_tfidf, predictions_sm))
print("Accuracy : ",accuracy_score(y_test_tfidf, predictions_sm))
print("F1 score: ",f1_score(y_test_tfidf, predictions_sm))
print("Recall: ",recall_score(y_test_tfidf, predictions_sm))
print("Precision: ",precision_score(y_test_tfidf, predictions_sm))


# In[34]:


# Saving the model as it will be used later while deploying
import pickle

# Save to file in the current working directory
pkl_filename = "LR_clf_final_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lr_sm, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickled_model = pickle.load(file)


# In[35]:


#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=300)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_sm,y_train_sm)

y_pred_RF =clf.predict(X_test_tfidf)

confusion_RF = confusion_matrix(y_test_tfidf, y_pred_RF)
print(confusion_RF)

# print classification report

print(classification_report(y_test_tfidf, y_pred_RF))
print("F1 score: ",f1_score(y_test_tfidf, y_pred_RF))


# In[36]:


# Create the parameter grid based on the results of random search 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [10,20,30],
    'min_samples_leaf': [100,125,150,175],
    'min_samples_split': [200,250,300],
    'n_estimators': [250,350,500], 
    'max_features': [10,15]
}
# Create a based model
rf_grid = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf_grid, param_grid = param_grid, cv = 3, scoring="recall", n_jobs = -1,verbose = 1)


# In[37]:



# Fit the grid search to the data
grid_search.fit(X_train_sm,y_train_sm)


# In[38]:


# printing the optimal accuracy score and hyperparameters
print('We can get recall of',grid_search.best_score_,'using',grid_search.best_params_)


# In[39]:


# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=30,
                             min_samples_leaf=150, 
                             min_samples_split=300,
                             max_features=10,
                             n_estimators=500)


# In[40]:


# fit
rfc.fit(X_train_sm,y_train_sm)

pred_hyperP_RF = rfc.predict(X_test_tfidf)
# Confusion matrix 
confusion_hyperP_RF = confusion_matrix(y_test_tfidf,pred_hyperP_RF)
print(confusion_hyperP_RF)


# In[41]:


# print classification report
print(classification_report(y_test_tfidf, pred_hyperP_RF))
print("Accuracy : ",accuracy_score(y_test_tfidf, pred_hyperP_RF))
print("F1 score: ",f1_score(y_test_tfidf, pred_hyperP_RF))
print("Recall: ",recall_score(y_test_tfidf, pred_hyperP_RF))
print("Precision: ",precision_score(y_test_tfidf, pred_hyperP_RF))


# In[42]:


# fit model on training data
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train_sm,y_train_sm)


# In[43]:


y_pred_xgboost = model.predict(X_test_tfidf)
pred_xgboost = [round(value) for value in y_pred_xgboost]

confusion_XGBoost = confusion_matrix(y_test_tfidf, pred_xgboost)
print(confusion_XGBoost)


# In[44]:


# print classification report
print(classification_report(y_test_tfidf, pred_xgboost))

print("Accuracy : ",accuracy_score(y_test_tfidf, pred_xgboost))
print("F1 score: ",f1_score(y_test_tfidf, pred_xgboost))
print("Recall: ",recall_score(y_test_tfidf, pred_xgboost))
print("Precision: ",precision_score(y_test_tfidf, pred_xgboost))


# In[45]:


from sklearn.naive_bayes import MultinomialNB

MNB=MultinomialNB()
MNB.fit(X_train_sm,y_train_sm)


# In[46]:


pred_NB=MNB.predict(X_test_tfidf)

confusion_NB = confusion_matrix(y_test_tfidf, pred_NB)
print(confusion_NB)


# In[47]:


# print classification report
print(classification_report(y_test_tfidf, pred_NB))

print("Accuracy : ",accuracy_score(y_test_tfidf, pred_NB))
print("F1 score: ",f1_score(y_test_tfidf, pred_NB))
print("Recall: ",recall_score(y_test_tfidf, pred_NB))
print("Precision: ",precision_score(y_test_tfidf, pred_NB))


# In[48]:


# Deleting rows where username is null

User_recom= df2[df2['reviews_username'].isnull()== False]
User_recom.reset_index(drop=True)
print(len(User_recom))
User_recom.head(5)


# In[49]:


# Test and Train split of the dataset.
train_ur, test_ur = train_test_split(User_recom, test_size=0.30, random_state=100)


# In[50]:


# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
df_pivot = train_ur.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(0)


# In[51]:


df_pivot.head(5)


# Next we will create following datasets which will be used for prediction
# 
# Dummy train will be used later for prediction of the products which have not been rated by the user. To ignore the products rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset.
# 
# Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

# In[52]:



# Copy the train dataset into dummy_train
dummy_train = train_ur.copy()

# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[53]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)

dummy_train.head(5)


# In[54]:



from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_corr = 1 - pairwise_distances(df_pivot, metric='cosine')
user_corr[np.isnan(user_corr)] = 0
print(user_corr)


# In[55]:


user_corr[user_corr<0]=0
user_corr


# In[56]:


user_pred_ratings = np.dot(user_corr, df_pivot.fillna(0))
user_pred_ratings


# In[57]:


user_pred_ratings.shape


# In[58]:


user_rating = np.multiply(user_pred_ratings,dummy_train)
user_rating.head()


# In[59]:


# Take the user name as input.
user_input = str(input("Enter your user name"))
print(user_input)


# In[60]:


d = user_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


# In[61]:



mapping=User_recom[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)
mapping.head()


# In[62]:


# Merging product id with mapping file to get the name of the recommended product
d = pd.merge(d,mapping, left_on='id', right_on='id', how = 'left')
d


# In[63]:


# Find out the common users of test and train dataset.
common_user = test_ur[test_ur.reviews_username.isin(train_ur.reviews_username)]


# In[64]:


common_user.head()


# In[65]:


# convert into the user-product matrix.
common_user_matrix = common_user.pivot_table(index='reviews_username',columns='id',values='reviews_rating')
common_user_matrix.head()


# In[66]:



# Convert the user_correlation matrix into dataframe.
user_corr = pd.DataFrame(user_corr)
user_corr.head()


# In[67]:



user_corr['reviews_username'] = df_pivot.index

user_corr.set_index('reviews_username',inplace=True)
user_corr.head()


# In[68]:



list_name = common_user.reviews_username.tolist()

user_corr.columns = df_pivot.index.tolist()

user_corr1 =  user_corr[user_corr.index.isin(list_name)]


# In[69]:



user_corr1.shape


# In[70]:


user_corr2 = user_corr1.T[user_corr1.T.index.isin(list_name)]


# In[71]:


user_corr3 = user_corr2.T


# In[72]:


print(user_corr3.shape)
user_corr3.head()


# In[73]:


user_corr3[user_corr3<0]=0

common_user_ratings = np.dot(user_corr3, common_user_matrix.fillna(0))
common_user_ratings


# In[74]:



dummy_test = common_user.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username',columns='id',values='reviews_rating').fillna(0)
print(dummy_test.shape)


# In[75]:


print(common_user_matrix.shape)
common_user_matrix.head()


# In[76]:



print(dummy_test.shape)
dummy_test.head()


# In[77]:


common_user_ratings = np.multiply(common_user_ratings,dummy_test)
common_user_ratings.head()


# In[78]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))
print(y)


# In[79]:


common_1 = common_user.pivot_table(index='reviews_username',columns='id',values='reviews_rating')

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_1 - y )**2))/total_non_nan)**0.5
print(rmse)


# Item based 

# In[80]:



df_pivot = train_ur.pivot_table(
   index='reviews_username',
    columns='id',
    values='reviews_rating'
).T

df_pivot.head()


# In[81]:


from sklearn.metrics.pairwise import pairwise_distances

# Item Similarity Matrix
item_corr = 1 - pairwise_distances(df_pivot.fillna(0), metric='cosine')
item_corr[np.isnan(item_corr)] = 0
print(item_corr)


# In[82]:


item_corr.shape


# In[83]:


item_ratings = np.dot((df_pivot.fillna(0).T),item_corr)
item_ratings


# In[84]:


print(item_ratings.shape)
print(dummy_train.shape)


# In[85]:


item_final_rating = np.multiply(item_ratings,dummy_train)
item_final_rating.head()


# In[86]:


# Take the user ID as input
user_input = str(input("Enter your user name"))
print(user_input)


# In[87]:



# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


# In[88]:



mapping_item= User_recom[['id','name']]
mapping_item = pd.DataFrame.drop_duplicates(mapping_item)
mapping_item.head()


# In[89]:


d = pd.merge(d,mapping_item, left_on='id', right_on='id', how = 'left')
d


# Evaluation Item based

# In[90]:


test_ur.columns


# In[91]:


common_test_item = test_ur[test_ur.id.isin(train_ur.id	)]
print(common_test_item .shape)
common_test_item .head()


# In[92]:


common_item_matrix = common_test_item .pivot_table(index='reviews_username', columns='id', values='reviews_rating').T


# In[93]:


common_item_matrix.shape


# In[94]:


item_corr = pd.DataFrame(item_corr)
item_corr.head(5)


# In[95]:


item_corr['id'] = df_pivot.index
item_corr.set_index('id',inplace=True)
item_corr.head()


# In[96]:


list_name = common_test_item.id.tolist()

item_corr.columns = df_pivot.index.tolist()

item_corr1 =  item_corr[item_corr.index.isin(list_name)]

item_corr2 = item_corr1.T[item_corr1.T.index.isin(list_name)]

item_corr3 = item_corr2.T

item_corr3.head()


# In[97]:



item_corr3[item_corr3<0]=0
common_item_ratings = np.dot(item_corr3, common_item_matrix.fillna(0))
print(common_item_ratings.shape)
common_item_ratings


# In[98]:



dummy_test = common_test_item.copy()
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = dummy_test.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T.fillna(0)
common_item_ratings = np.multiply(common_item_ratings,dummy_test)


# In[99]:



# The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.

common_2 = common_test_item.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[100]:



# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_2 - y )**2))/total_non_nan)**0.5
print(rmse)


# Conclusion

# In[101]:


# Take the user ID as input
user_input = str(input("Enter your user name"))
print(user_input)


# In[102]:


recommendations = pd.DataFrame(user_rating.loc[user_input].sort_values(ascending=False)[0:20]).reset_index()
mapping= User_recom[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)
recommendations = pd.merge(recommendations,mapping, left_on='id', right_on='id', how = 'left')
recommendations


# In[103]:



import pickle

user_rating.to_pickle("user_rating.pkl")
pickled_user_rating = pd.read_pickle("user_rating.pkl")
pickled_user_rating


# In[104]:


# Save to file in the current working directory

mapping.to_pickle("product_name.pkl")
pickled_mapping = pd.read_pickle("product_name.pkl")
pickled_mapping


# In[105]:



# Save to file in the current working directory

df2.to_pickle("reviews_cols.pkl")
pickled_reviews_data = pd.read_pickle("reviews_cols.pkl")
pickled_reviews_data


# In[106]:


df2['total_review'].apply(str)


# 
# 3. Improving the recommendations using the sentiment analysis model¶
# Fine-Tuning the recommendation system and recommending top 5 products to the user based on highest percentage of positive sentiments using Sentiment Analysis model developed earlier

# In[107]:


# Predicting sentiment for the recommended products using the Logistic Regression model developed earlier

prod_recommendations= pd.merge(recommendations,pickled_reviews_data[['id','reviews_clean_text']], left_on='id', right_on='id', how = 'left')
test_data_for_user = pickled_tfidf_vectorizer.transform(prod_recommendations['reviews_clean_text'])
sentiment_prediction_for_user= pickled_model.predict(test_data_for_user)
sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])
prod_recommendations= pd.concat([prod_recommendations, sentiment_prediction_for_user], axis=1)


# In[108]:


# For each of the 20 recommended products, calculating the percentage of positive sentiments 
#   for all the reviews of each product

a=prod_recommendations.groupby('id')
b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
b.columns = ['id', 'Total_reviews']
c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
c.columns = ['id', 'Total_predicted_positive_reviews']
prod_recommendations_final=pd.merge( b, c, left_on='id', right_on='id', how='left')
prod_recommendations_final['Positive_sentiment_rate'] = prod_recommendations_final['Total_predicted_positive_reviews'].div(prod_recommendations_final['Total_reviews']).replace(np.inf, 0)
prod_recommendations_final= prod_recommendations_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
prod_recommendations_final=pd.merge(prod_recommendations_final, pickled_mapping, left_on='id', right_on='id', how='left')

# Filtering out the top 5 products with the highest percentage of positive review
prod_recommendations_final.head(5)


# In[ ]:




