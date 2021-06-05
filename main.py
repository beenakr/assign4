import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option("deprecation.showPyplotGlobalUse", False)

heart = pd.read_csv("heart.csv")

st.title("EDA for Heart Attack Dataset")
st.write(
    "Link: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset"
)
st.write("There are two dataset heart.csv and o2Saturation.csv")
st.text("For this EDA we are only going yo look at heart.csv")

st.header("About dataset")
st.markdown(
    """
    - age : Age of the patient
    - sex : Sex of the patient
    - cp : Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
    - trtbps : Resting blood pressure (in mm Hg)
    - chol : Cholestoral in mg/dl fetched via BMI sensor
    - fbs : (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False
    - restecg : Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy
    - thalachh : Maximum heart rate achieved
    - oldpeak : Previous peak
    - slp : Slope
    - caa : Number of major vessels
    - thall : Thalium Stress Test result ~ (0,3)
    - exng : Exercise induced angina ~ 1 = Yes, 0 = No
    - output : target : 0= less chance of heart attack 1= more chance of heart attack
"""
)

st.subheader("Dataset Head")
st.dataframe(heart.head())
st.subheader("Dataset Description")
st.dataframe(heart.describe())
st.subheader("Number of unique data")
st.dataframe(heart.nunique())
st.subheader("Checking missing values")
st.dataframe(heart.isnull().sum())

st.title("Univariate Analysis")
st.header("Categorical and Target features")
heart["output"].value_counts(normalize=True).plot.bar(
    color=["red", "blue"], edgecolor="black", title="target variable"
)
st.pyplot()
st.markdown(
    """
    - Around 55% people have more chances to get heart attack
    - Around 45% people have less chances to get heart attack"""
)
st.subheader("Sex features")
heart["sex"].value_counts(normalize=True).plot.bar(
    color=["cyan", "magenta"], edgecolor="black", title="sex variable"
)
st.pyplot()
st.markdown(
    """
    - Around 68 % people are with sex=1
    - Around 30 % people are with sex=0"""
)
st.subheader("Chest pain features")
heart["cp"].value_counts(normalize=True).plot.bar(
    color=["yellow", "orange", "cyan", "magenta"],
    edgecolor="black",
    title="chest pain variable",
)
st.pyplot()
st.markdown(
    """
    - Around 50 % of the people have chest pain type: Typical Angina
    - Around 28 % of the people have chest pain type: Non-anginal Pain
    - Around less than 20 % of the people have chest pain type: Atypical Angina
    - Around less than 10% of the people have chest pain type: Asymptomatic"""
)

st.subheader(
    "1.exercise induced angina, 2.fasting blood sugar > 120 mg/dl, 3.resting electrocardiographic results, 4. Slope"
)
plt.figure(figsize=(20, 7))
plt.subplot(221)
heart["exng"].value_counts(normalize=True).plot.bar(
    color=["yellow", "orange"],
    edgecolor="black",
    title="exercise induced angina (1 = yes; 0 = no)",
)
plt.subplot(222)
heart["fbs"].value_counts(normalize=True).plot.bar(
    color=["yellow", "green"],
    edgecolor="black",
    title="(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)",
)
plt.subplot(223)
heart["restecg"].value_counts(normalize=True).plot.bar(
    color=["magenta", "blue", "cyan"],
    edgecolor="black",
    title="resting electrocardiographic results",
)
plt.subplot(224)
heart["slp"].value_counts(normalize=True).plot.bar(
    color=["red", "blue", "green"], edgecolor="black", title="- Slope"
)
st.pyplot(plt)
st.markdown(
    """
    - More than 65 % of the people Exercise don't induced angina
    - More than 35 % of the people Exercise induced angina
    - less than 20 % of the people have fasting blood sugar > 120 mg/dl
    - More than 80 % of the people have fasting blood sugar <= 120 mg/dl
    - less than 50 % of the people have resting electrocardiographic results normal
    - 50 % of the people have resting electrocardiographic results: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - 1% or 2% of the people have resting electrocardiographic results: showing probable or definite left ventricular hypertrophy by Estes' criteria
    """
)

st.subheader("1.number of major vessels, 2.Thalium Stress Test result ~ (0,3)")
plt.figure(figsize=(20, 7))
plt.subplot(221)
heart["caa"].value_counts(normalize=True).plot.bar(
    color=["magenta", "blue", "cyan", "red", "orange"],
    edgecolor="black",
    title="number of major vessels",
)
plt.subplot(222)
heart["thall"].value_counts(normalize=True).plot.bar(
    color=["lightblue", "lightgreen", "lightyellow", "magenta"],
    edgecolor="black",
    title="Thalium Stress Test result ~ (0,3)",
)
st.pyplot(plt)

st.header("Numerical features")
st.subheader("age , blood pressure , cholestoral , Heart Rate")
plt.figure(figsize=(20, 7))
plt.subplot(221)
heart["age"].plot.hist(edgecolor="black", color="lightgreen", title="age variable")
plt.subplot(222)
heart["trtbps"].plot.hist(
    edgecolor="black", color="lightblue", title="resting blood pressure in mm hg"
)
plt.subplot(223)
heart["chol"].plot.hist(
    edgecolor="black",
    color="lightcoral",
    title="cholestoral in mg/dl fetched via BMI sensor",
)
plt.subplot(224)
heart["thalachh"].plot.hist(
    edgecolor="black", color="lightgray", title="maximum heart rate achieved"
)
st.pyplot(plt)

st.subheader("Oldpeak")
heart["oldpeak"].plot.hist(
    edgecolor="black", color="lightyellow", title="oldpeak variable"
)
st.pyplot()

st.title("Bivariate Analysis")
st.subheader("effect of age on heart attack")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on heart attack")
sns.lineplot(x=heart["age"], y=heart["output"])
st.pyplot()
st.markdown(
    """
    - The people with the age 30 to 35 have higher chance of heart attacks
    - The people with the age than 70 and less than 75 have higher chance of heart attacks
    - apart from it no certain trend i will be able to find
    """
)

st.subheader("heart attack related with sex")
sns.countplot(data=heart, x="sex", palette=["blue", "red"], hue="output")
st.pyplot()
st.markdown(
    """
    - people of sex=1 have higher chances of getting heart attacks
    """
)

st.subheader("heart attack related with sex")
sns.kdeplot(data=heart, x="cp", hue="output", fill=True)
st.pyplot()
st.markdown(
    """
    - people with chest pain type=2 have higher chance of getting heart attacks
    """
)

st.subheader("heart attack related with age")
sns.kdeplot(data=heart, x="age", hue="output", fill=True)
st.pyplot()
st.markdown(
    """
    - according to the data people with lower age have more chances of getting heart attacks than those of higher ages
    """
)

st.subheader("heat attack realted with thalium stress test")
sns.kdeplot(data=heart, x="thall", hue="output", fill=True)
st.pyplot()
st.markdown(
    """
    - people with thall test=2 have higher chance of getting heart attacks
    """
)

st.subheader("heat attack realted with Exercise induced angina")
sns.kdeplot(data=heart, x="exng", hue="output", fill=True)
st.pyplot()
st.markdown(
    """
    - people with exng=0 have higher chances of getting heart attacks
    """
)

st.subheader("effect of age on blood pressure")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on blood pressure")
sns.lineplot(x=heart["age"], y=heart["trtbps"])
st.pyplot()
st.markdown(
    """
    - as age is incresing the increase in the blood pressure has been founded
    """
)

st.subheader("effect of age on cholestrol level")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on cholestrol level")
sns.lineplot(x=heart["age"], y=heart["chol"])
st.pyplot()
st.markdown(
    """
    - as age is incresing the increase in the cholestrol level has been founded
    """
)

st.subheader("effect of age on heart rate")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on heart rate")
sns.lineplot(x=heart["age"], y=heart["thalachh"])
st.pyplot()
st.markdown(
    """
    - as age is incresing the decrease in the heart rate has been founded
    """
)

st.subheader("How does incresed heart rate and age affect the heart attack")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of heart attack with increase in age and heart rate")
sns.lineplot(x=heart["age"], y=heart["thalachh"], hue=heart["output"])
st.pyplot()
st.markdown(
    """
    - as with the increase in the age the heart rate is decresing and also the people with more chances of heart attacks are decreasing hence we can say higher heart rate increases the chance of heart attack
    """
)
sns.kdeplot(data=heart, x="thalachh", hue="output", fill=True)
st.pyplot()

st.subheader("How does incresed cholestrol and age affect the heart attack")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of heart attack with increase in age and cholestrol")
sns.lineplot(x=heart["age"], y=heart["chol"], hue=heart["output"])
st.pyplot()
st.markdown(
    """
    - as with the increase in the age the cholestrol level is incresing and also the people with more chances of heart attacks are also increasing hence we can say higher cholestrol level increases the chance of heart attack
    """
)
sns.kdeplot(data=heart, x="chol", hue="output", fill=True)
st.pyplot()

st.subheader("How does incresed blood pressure and age affect the heart attack")
plt.figure(figsize=(10, 7))
plt.style.use("fivethirtyeight")
plt.title("effect of heart attack with increase in age and blood pressure")
sns.lineplot(x=heart["age"], y=heart["trtbps"], hue=heart["output"])
st.pyplot()
st.markdown(
    """
    - as with the increase in the age the blood pressure is incresing and also the people with more chances of heart attacks are also increasing hence we can say blood pressure increases the chance of heart attack
    """
)
sns.kdeplot(data=heart, x="trtbps", hue="output", fill=True)
st.pyplot()
