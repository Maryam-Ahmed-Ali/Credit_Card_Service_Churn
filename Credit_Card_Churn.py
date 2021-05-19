import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(layout='wide')
st.title("Credit Card Service Churn 💳")
st.sidebar.title("Credit Card Service Churn 💳")
"""
The data is about credit card service clients in a bank to identify the types of clients mentioning their age,
salary, credit card limit, etc, and who attrited among them in order to develop marketing strategies
based on data insights and machine learning models to know who are more likely to churn so they can take actions accordingly
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
"""

#@st.cache
# def loading_data():
#     data =pd.read_csv("BankChurners.csv")
#     data = data.iloc[:,1:]
#     return data

data = pd.read_csv("BankChurners.csv")
data.head()

data.info()

#Exploring outliers
def count_outliers(col):
    count = 0
    q1 = np.nanpercentile(col, 25)
    q3 = np.nanpercentile(col, 75)
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    for i in col.values:
      if i > upper or i < lower:
        count += 1
    return count

#Exploring percentage of outliers in numerical columns
num_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
for i in num_vars:
    print (count_outliers(data[i])/len(data))


#Exploring categorical data
for feature in data.columns:
    if data[feature].dtype not in ['int64', 'float64']:
        print(f'{feature}: {data[feature].unique()}')

#Show raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(data)

#beta_columns
chart1, chart2 = st.beta_columns(2)
#Number of crdeit cards issued
with chart1:
    product = data['Card_Category'].unique()
    product = np.append(product, 'All')
    option = st.selectbox("What product do you want to know about?", product)
    if option == 'All':
        num = len(data)
    else:
        num = len(data[data['Card_Category']==option])
    if option == 'All':
        st.markdown(
            f'<div><p style = "color: rgb(23,23,23); font-size:35px">We issued {num} of credit cards in total</div>',
            unsafe_allow_html=True)

    else:
        st.markdown(
            f'<div><p style = "color: rgb(23,23,23); font-size:35px">We issued {num} {option} credit cards</div>',
            unsafe_allow_html=True)



#Number of churned vs unchurned customers in each Product
with chart2:
    st.subheader(f'Existing vs Attrited clients for {option}')
    if option == 'All':
        cards_ch = data['Attrition_Flag'].value_counts(normalize=True)
        fig = px.pie(data, values=cards_ch, names=cards_ch.index)
        fig.update_layout(width=450,height=350)
        st.plotly_chart(fig)
    else:
        cards_ch = data.loc[data['Card_Category']==option,'Attrition_Flag'].value_counts(normalize=True)
        fig = px.pie(data, values=cards_ch, names=cards_ch.index)
        fig.update_layout(width=450,height=350)
        st.plotly_chart(fig)

chart3, chart4, chart5 = st.beta_columns(3)
#Number of credit cards issued by gender
with chart3:
    st.subheader(f'Clients by gender for {option}')
    if option == 'All':
        fig_g = px.histogram(
                data,
                x=data['Gender'],
                color=data['Gender'])
        fig_g.update_layout(width=300,height=350)
        st.plotly_chart(fig_g)

    else:
        cards_g = data.loc[data['Card_Category']==option,'Gender']
        fig_g = px.histogram(
            data,
            x=cards_g,
            color=cards_g)
        fig_g.update_layout(width=300,height=350)
        st.plotly_chart(fig_g)


#Number of credit cards issued by marital status
with chart4:
    st.subheader(f'Clients by marital status for {option}')
    if option == 'All':
        fig_m = px.histogram(
                data,
                x=data['Marital_Status'],
                color=data['Marital_Status'])
        fig_m.update_layout(width=300,height=350)
        st.plotly_chart(fig_m)

    else:
        cards_m = data.loc[data['Card_Category']==option,'Marital_Status']
        fig_m = px.histogram(
            data,
            x=cards_m,
            color=cards_m)
        fig_m.update_layout(width=300,height=350)
        st.plotly_chart(fig_m)


#Number of credit cards issued by income category of clients
with chart5:
    st.subheader(f'Clients by income category for {option}')
    if option == 'All':
        fig_income_c = px.histogram(
                data,
                x=data['Income_Category'],
                color=data['Income_Category'])
        fig_income_c.update_layout(width=400,height=350)
        st.plotly_chart(fig_income_c)

    else:
        cards_income_c = data.loc[data['Card_Category']==option,'Income_Category']
        fig_income_c = px.histogram(
            data,
            x=cards_income_c,
            color=cards_income_c)
        fig_income_c.update_layout(width=400,height=350)
        st.plotly_chart(fig_income_c)


chart6, chart7 = st.beta_columns(2)
#Number of credit cards issued by education level of clients
with chart6:
    st.subheader(f'Clients by education level for {option}')
    if option == 'All':
        fig_education = px.histogram(
                data,
                x=data['Education_Level'],
                color=data['Education_Level'])
        fig_education.update_layout(width=400,height=350)
        st.plotly_chart(fig_education)

    else:
        cards_education = data.loc[data['Card_Category']==option,'Education_Level']
        fig_education = px.histogram(
            data,
            x=cards_education,
            color=cards_education)
        fig_education.update_layout(width=400,height=350)
        st.plotly_chart(fig_education)


#How total transactions count spread across age
with chart7:
    st.subheader('How credit cards\' count of total transactions spread across clients\' age')
    if option == 'All':
        data = data
    else:
        data = data[data['Card_Category']==option]
    sc = px.scatter(data, x='Customer_Age', y='Total_Trans_Ct',width=400, height=350)
    st.plotly_chart(sc, width=400,height=350)


#Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


#Data preprocessing
#@st.cache
def split(df):
    y = df.Attrition_Flag
    y = y.replace({'Existing Customer':0,
                   'Attrited Customer':1})
    x = df.drop(columns=['Attrition_Flag'])
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, stratify= y, random_state=0)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(data)
cat_vars = x_train.select_dtypes(include=['object']).columns.tolist()
num_vars = x_train.select_dtypes(include=['float', 'int']).columns.tolist()

scaler = StandardScaler()
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

full_pipeline = ColumnTransformer(
    transformers = [
        ('cat', ohe, cat_vars),
        ('num', scaler, num_vars)],
)

x_train= full_pipeline.fit_transform(x_train)
x_test= full_pipeline.transform(x_test)

#Classification
st.sidebar.subheader('Choose Classifier')
classifier = st.sidebar.selectbox('Classifier',('Logistic Regression', 'Random Forest','GBM'))

st.set_option('deprecation.showPyplotGlobalUse', False)
#Plotting evaluation metrics
def plot_metrics(metrics_list):
    if 'Confusion Matrix'  in metrics_list:
        st.subheader('Confusion Matrix')
        st.write(confusion_matrix(y_test, y_pred1))

    if 'Classification Report' in metrics_list:
        st.subheader('Classification Report')
        st.write(pd.DataFrame(classification_report(y_test, y_pred1, output_dict=True)).T)
        st.pyplot()


#Building and testing models
if classifier == 'Logistic Regression':
    metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'Classification Report'))
    if st.sidebar.button('Classify', key = 'classify'):
        st.subheader('Logistic Regression Results')
        model = LogisticRegression(C=10, max_iter = 500)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict_proba(x_test)[:, 1]
        y_pred1 = np.ones((len(y_pred),), dtype=int)
        for i in range(len(y_pred)):
            if y_pred[i] <= 0.2:
                y_pred1[i]=0
        else:
                y_pred1[i]=1
        st.write('Accuracy', accuracy.round(2))
        st.write('AUC', roc_auc_score(y_test, y_pred).round(2))
        plot_metrics(metrics)


if classifier == 'Random Forest':
    metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'Classification Report'))
    if st.sidebar.button('Classify', key = 'classify'):
        st.subheader('Random Forest Results')
        model = RandomForestClassifier(n_estimators=500, max_depth=20, bootstrap='False')
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict_proba(x_test)[:, 1]
        y_pred1 = np.ones((len(y_pred),), dtype=int)
        for i in range(len(y_pred)):
            if y_pred[i] <= 0.2:
                y_pred1[i]=0
        else:
                y_pred1[i]=1
        st.write('Accuracy', accuracy.round(2))
        st.write('AUC', roc_auc_score(y_test, y_pred).round(2))
        plot_metrics(metrics)

if classifier == 'GBM':
    metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'Classification Report'))
    if st.sidebar.button('Classify', key = 'classify'):
        st.subheader('GBM Results')
        model = GradientBoostingClassifier(learning_rate=0.5, n_estimators=500)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict_proba(x_test)[:, 1]
        y_pred1 = np.ones((len(y_pred),), dtype=int)
        for i in range(len(y_pred)):
            if y_pred[i] <= 0.2:
                y_pred1[i]=0
        else:
                y_pred1[i]=1
        st.write('Accuracy', accuracy.round(2))
        st.write('AUC', roc_auc_score(y_test, y_pred).round(2))
        plot_metrics(metrics)
