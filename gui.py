import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/refs/heads/main/inliers0.01.csv"
df = pd.read_csv(data_url, header=0)

# Encode categorical variable
le = LabelEncoder()
df['Fiber Type'] = le.fit_transform(df['Fiber Type'])

# Split features and target
x = df.iloc[:,:-1].values  # Convert to numpy array
y = df.iloc[:,-1].values   # Convert to numpy array
scaler=StandardScaler()
x=scaler.fit_transform(x)

et_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/main/et_model.pkl"
lgbm_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/main/lgbm_model.pkl"
xgb_url="https://raw.githubusercontent.com/cakirogl/splitting_tensile_composite/main/xgb_model.pkl"
response = requests.get(et_url)
#et_model = pickle.loads(response.content)
et_model = ExtraTreesRegressor()
et_model.fit(x,y)

ic=st.container()
ic1,ic2 = ic.columns(2)
with ic1:
    cement = st.number_input("**Cement [$kg/m^3$]**", min_value=243.0, max_value=810.0, step=3.0, value=452.0);
    WB = st.number_input("**Water/Binder Ratio**", min_value=0.26, max_value=0.75, step=0.02, value=0.4)
    FA = st.number_input("**Fine Aggregate**", min_value=470.3, max_value=1065.0, step=10.0, value=600.0)
    CA = st.number_input("**Coarse Aggregate**", min_value=645.0, max_value=1420.0, step=10.0, value=1200.0)
    RCA = st.number_input("**Recycled Coarse Aggregate [\%]**", min_value=0.0, max_value=100.0, step=10.0, value=20.0)
    SCM = st.number_input("**Supplementary Cementitious Materials [$kg/m^3$]**", min_value=0.0, max_value=153.9, step=10.0, value=0.0)
with ic2:
    SP = st.number_input("**Superplasticizer [$kg/m^3$]**", min_value=0.0, max_value=20.5, step=2.5, value=0.0)
    NFP = st.number_input("**Natural Fiber Percentage**", min_value=0.0, max_value=3.0, step=1.0, value=0.0)
    FT = st.selectbox("**Fiber Type**", ["None", "Jute", "Kenaf", "Bamboo", "Sisal", "Coir", "Ramie"])
    if FT=="None":
        FT=0.0
    elif FT == "Jute":
        FT=1.0
    elif FT == "Kenaf":
        FT = 2.0
    elif FT == "Bamboo":
        FT = 3.0
    elif FT == "Sisal":
        FT = 4.0
    elif FT == "Coir":
        FT =5.0
    elif FT == "Ramie":
        FT=6.0
    L = st.number_input("**Fiber Length [$mm$]**", min_value=0.0, max_value=60.0, step=5.0, value=0.0)
    Age = st.number_input("**Age [$days$]**", min_value=1.0, max_value=345.0, step=5.0, value=28.0) 
oc=st.container()
new_sample = np.array([[cement, WB, FA, CA, RCA, SCM, SP, NFP, FT, L, Age]], dtype=object)
with ic2:
    st.write(f":blue[**The tensile strength = **{et_model.predict(new_sample)[0]:.2f}** [MPa]**]")