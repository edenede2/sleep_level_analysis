import streamlit as st
import pandas as pd
import plotly.express as px



# Title of the web app
st.title('Excel Data Analysis Web App')

# Sidebar for upload
st.sidebar.header("Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Reading the uploaded excel file
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Showing the dataframe
    st.write("Data Preview:")
    st.write(df.head())

    # Filtering subjects
    st.sidebar.header("Filter Subjects")
    unique_subjects = df['Id'].unique()
    subjects_to_analyze = st.sidebar.multiselect('Select Subjects', unique_subjects, default=unique_subjects)
    
    # Filtering dataframe based on selected subjects
    df_filtered = df[df['Id'].isin(subjects_to_analyze)]
    
    # Select pairs of columns
    st.sidebar.header("Select Columns to Analyze")
    columns = df.columns.tolist()
    x_axis = st.sidebar.selectbox('Select X-axis', options=columns, index=columns.index('dayOfExperiment') if 'dayOfExperiment' in columns else 0)
    y_axis = st.sidebar.selectbox('Select Y-axis', options=columns, index=1)

    # Plotting
    if st.sidebar.button("Generate Plot"):
        fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color='Id', title=f'{y_axis} vs {x_axis}')
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    st.sidebar.header("Correlation Analysis")
    if st.sidebar.button("Show Correlation Matrix"):
        corr_matrix = df_filtered.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

# Running Instructions
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload your Excel file.\n2. Filter subjects if needed.\n3. Select columns for analysis.\n4. Generate plots and view correlation matrix.")
