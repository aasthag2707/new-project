# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:20:35 2021

@author: deepak
"""

import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('user_offer_attraction.pkl', 'rb'))
dataset= pd.read_csv('marketing_dataset.csv')
X=dataset.drop(["Response","Income","Total Spent","Total Purchase","Recency","Complain","Teenhome"], axis = 1)

y=dataset[['Response']]
y=np.array(y,dtype='int64').ravel()

encoder=ColumnTransformer([('encoder',OneHotEncoder(), [0,1,19])],remainder='passthrough')
X=encoder.fit_transform(X)

temp1=X[:, 1:5]
temp2=X[:, 6:13]
temp3=X[:, 14:]

X=np.concatenate((temp1,temp2,temp3), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=27)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)



def predict_note_authentication(Education,Marital_Status,Kidhome,MntWines,MntFruits,MntMeatProducts,MntFishProducts,	MntSweetProducts,
MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,
AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Country):

  X_ui=encoder.transform([[Education,Marital_Status,Kidhome,MntWines,MntFruits,MntMeatProducts,MntFishProducts,	MntSweetProducts,
  MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,
  AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Country]])

  temp1=X_ui[:, 1:5]
  temp2=X_ui[:, 6:13]
  temp3=X_ui[:, 14:]

  X_ui=np.concatenate((temp1,temp2,temp3), axis=1)

  scaled_data= sc.transform(X_ui)


  out=pca.transform(scaled_data)
  output=model.predict(out)

  print("Response", output)

  if output==[1]:
    prediction="customer accepted the offer in the campaign"
  else:
    prediction="customer rejected the offer in the campaign"
  print(prediction)
  return prediction

def main():

    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction using SVM Algorithm")



    Education = st.selectbox(
    "Education",
    ("Graduation","PhD","Master","2n Cycle","Basic"))

    Marital_Status = st.selectbox(
    "Marital_Status",
    ("Married","Together","Single","Divorced","Widow","Alone","YOLO","Absurd"))

    Kidhome= st.number_input('Insert KidhomE',0,2)
    MntWines = st.number_input('Insert MntWines',0,1493)
    MntFruits= st.number_input('Insert MntFruits',0,200)
    MntMeatProducts = st.number_input('Insert MntMeatProducts',0,1725)
    MntFishProducts = st.number_input('Insert MntFishProducts',0,259)
    MntSweetProducts = st.number_input('Insert MntSweetProducts',0,263)
    MntGoldProds = st.number_input('Insert MntGoldProds',0,365)
    NumDealsPurchases = st.number_input('Insert NumDealsPurchases',0,16)
    NumWebPurchases = st.number_input('Insert NumWebPurchases',0,27)
    NumCatalogPurchases = st.number_input('Insert NumCatalogPurchases',0,30)
    NumStorePurchases = st.number_input('Insert NumStorePurchases',0,15)
    NumWebVisitsMonth = st.number_input('Insert NumWebVisitsMonth',0,20)
    AcceptedCmp3 = st.number_input('Insert AcceptedCmp3',0,1)
    AcceptedCmp4 = st.number_input('Insert AcceptedCmp4',0,1)
    AcceptedCmp5 = st.number_input('Insert AcceptedCmp5',0,1)
    AcceptedCmp1 = st.number_input('Insert AcceptedCmp1',0,1)
    AcceptedCmp2 = st.number_input('Insert AcceptedCmp2',0,1)

    Country = st.selectbox(
    "Country",
    ("SP","SA","CA","AUS","IND","GER","US","ME"))

    resul=""
    if st.button("Prediction"):

      result=predict_note_authentication(Education,Marital_Status,Kidhome,MntWines,MntFruits,MntMeatProducts,MntFishProducts,	MntSweetProducts,
      MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,
      AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Country)

      st.success('KNN Model has predicted {}'.format(result))

    if st.button("About"):
      st.header("Developed by Team_6")
      st.subheader("Poornima")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment 5: Support Vector Machine and Random Forest</p></center>
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
