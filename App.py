#!/usr/bin/env python
# coding: utf-8

# In[6]:


from flask import Flask, jsonify, request, render_template

import numpy as np
import pandas as pd
import pickle
import nltk

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        
        user_input=[str(x) for x in request.form.values()]
        user_input=user_input[0]
        #print(user_input)
        pkl_Tfidf_vectorizer = pickle.load(open('Tfidf_vectorizer.pkl','rb'))
        pkl_LR_clf_final_model = pickle.load(open('LR_clf_final_model.pkl','rb'))
        pickled_user_rating = pickle.load(open('user_rating.pkl','rb'))        
        product_name_map= pickle.load(open('product_name.pkl','rb')) 
        reviews_data = pickle.load(open('reviews_cols.pkl','rb'))
        prod_recommendations = pd.DataFrame(pickled_user_rating.loc[user_input]).reset_index()
        prod_recommendations.rename(columns={prod_recommendations.columns[1]: "user_pred_rating" }, inplace = True)
        prod_recommendations = prod_recommendations.sort_values(by='user_pred_rating', ascending=False)[0:20]
       
        prod_recommendations.rename(columns={prod_recommendations.columns[0]: "prod_id" }, inplace = True)
        product_name_map.rename(columns={product_name_map.columns[0]: "prod_id" }, inplace = True)  
        reviews_data.rename(columns={reviews_data.columns[0]: "prod_id" }, inplace = True)
    
        prod_recommendations = pd.merge(prod_recommendations,product_name_map, left_on="prod_id", right_on="prod_id", how = "left")
        
        prod_recommendations_final= pd.merge(prod_recommendations,reviews_data[['prod_id','reviews_clean_text']], left_on='prod_id', right_on='prod_id', how = 'left')
        test_data_for_user = pkl_Tfidf_vectorizer.transform(prod_recommendations_final['reviews_clean_text'].values.astype('U'))
        
        sentiment_prediction_for_user = pkl_LR_clf_final_model.predict(test_data_for_user)
        sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])

        prod_recommendations_final= pd.concat([prod_recommendations_final, sentiment_prediction_for_user], axis=1)
        
        a=prod_recommendations_final.groupby('prod_id')
        b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
        b.columns = ['prod_id', 'Total_reviews']        
        c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
        c.columns = ['prod_id', 'Total_predicted_positive_reviews']
        
        prod_recommendations_final_final=pd.merge( b, c, left_on='prod_id', right_on='prod_id', how='left')
        
        prod_recommendations_final_final['Positive_sentiment_rate'] = prod_recommendations_final_final['Total_predicted_positive_reviews'].div(prod_recommendations_final_final['Total_reviews']).replace(np.inf, 0)
        
        prod_recommendations_final_final= prod_recommendations_final_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
        prod_recommendations_final_final=pd.merge(prod_recommendations_final_final, product_name_map, left_on='prod_id', right_on='prod_id', how='left')
        
        name_display= prod_recommendations_final_final.head(5)
        name_display= name_display['name']
        
        output = name_display.to_list()
        output.insert(0,"***")
        output="***\t \t***".join(output)
        return render_template('index.html', prediction_text='Top 5 prod_recommendations are- {}'.format(output))
    else :
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




