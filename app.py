import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page title
st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")

st.title("🏠 House Price Prediction Dashboard")

# Load dataset
df = pd.read_csv("train.csv")

# Train model
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter House Details")

area = st.sidebar.slider(
    "Living Area (sq ft)",
    int(df['GrLivArea'].min()),
    int(df['GrLivArea'].max()),
    1500
)

bedrooms = st.sidebar.slider(
    "Number of Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Number of Bathrooms",
    1,
    5,
    2
)

# Prediction
new_house = pd.DataFrame({
    'GrLivArea': [area],
    'BedroomAbvGr': [bedrooms],
    'FullBath': [bathrooms]
})

price = model.predict(new_house)[0]

st.subheader(f"💰 Predicted House Price: ${price:,.2f}")

# Input chart
fig1 = go.Figure(
    data=[
        go.Bar(
            x=['Area', 'Bedrooms', 'Bathrooms'],
            y=[area, bedrooms, bathrooms]
        )
    ]
)
fig1.update_layout(title="Your House Details")

st.plotly_chart(fig1, use_container_width=True)

# Dataset scatter plot
fig2 = px.scatter(
    df,
    x='GrLivArea',
    y='SalePrice',
    title='Dataset Houses'
)

fig2.add_scatter(
    x=[area],
    y=[price],
    mode='markers+text',
    marker=dict(size=18, color='red', symbol='star'),
    text=['Your House'],
    textposition='top center'
)

st.plotly_chart(fig2, use_container_width=True)

# Price comparison
avg_price = df['SalePrice'].mean()

fig3 = go.Figure(
    data=[
        go.Bar(
            x=['Average Price', 'Predicted Price'],
            y=[avg_price, price]
        )
    ]
)
fig3.update_layout(title="Price Comparison")

st.plotly_chart(fig3, use_container_width=True)