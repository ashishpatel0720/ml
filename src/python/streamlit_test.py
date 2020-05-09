import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.write("# My First App")


row  = st.slider("Enter number of rows", min_value=1, max_value=50 , value=3)
col  = st.slider("Enter number of columns", min_value=1, max_value=50 , value=3)

df=pd.DataFrame(np.random.random([row,col]))


if st.checkbox("Show Full data"):
    'Full Data',df # also shows data
    # st.write(df)
else:
    'Showig Head',df.head(5) #otherwise show head


x = st.sidebar.slider("Select value of x", value = 2)

if( st.checkbox("show value of x")):
    'value of x:',x

plt.plot(np.sin(np.arange(-x,x,0.1)))
plt.plot(np.cos(np.arange(-x,x,0.1)))
plt.xlabel("value of x")
plt.ylabel("value of F(x)")
st.pyplot()

st.line_chart(np.sin(np.arange(-x,x,0.1)))
