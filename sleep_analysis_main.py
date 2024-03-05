import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np



# Custom function to evaluate the user-defined formula
def evaluate_formula(df, formula, param_name):
    # Extract column names from the formula
    columns = [col.strip() for col in formula.replace('+', ' ').replace('-', ' ').replace('*', ' ').replace('/', ' ').replace('(', ' ').replace(')', ' ').split()]
    columns = set(filter(lambda x: x in df.columns, columns))  # Keep only valid column names

    problematic_columns = []  # To store columns with types that cannot be converted to float
    for col in columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype(float)  # Convert int to float
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            problematic_columns.append((col, 'datetime'))
        elif df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            problematic_columns.append((col, 'string'))

    if problematic_columns:
        # Generate error message with problematic column names and their types
        error_message = "Error: The following columns have incompatible types: " + ", ".join([f"{name} ({dtype})" for name, dtype in problematic_columns])
        return False, error_message

    try:
        # Assuming df.eval can handle the formula after type checks and conversions
        df['custom_parameter'] = df.eval(formula)
        df.rename(columns={'custom_parameter': param_name}, inplace=True)
        return True, "Parameter added successfully!"
    except Exception as e:
        return False, str(e)

# Adjusted UI function to avoid direct modification after widget instantiation
def add_custom_parameter_ui(df):
    st.sidebar.header("Create Custom Parameter")
    all_columns = df.columns.tolist()

    # Separate state for managing formula building
    if 'formula_building' not in st.session_state:
        st.session_state.formula_building = ''

    # Display formula being built for user reference
    st.sidebar.text(f"Formula: {st.session_state.formula_building}")

    selected_parameter = st.sidebar.selectbox('Select Parameter', [''] + all_columns, index=0, format_func=lambda x: x if x else 'Select...')
    param_name = st.sidebar.text_input('Name for New Parameter', value='')

    # Button to append selected parameter to the building formula
    if st.sidebar.button('Append Parameter'):
        if selected_parameter:
            # Update the formula building state
            st.session_state.formula_building += f" {selected_parameter}"

    # User input for final formula modification or confirmation
    confirmed_formula = st.sidebar.text_input('Confirm or modify formula', value=st.session_state.formula_building, key='confirmed_formula')

    # Button to add the parameter
    if st.sidebar.button('Add Parameter'):
        if param_name and confirmed_formula:
            success, message = evaluate_formula(df, confirmed_formula, param_name)
            if success:
                st.success(message)
                # Reset formula building state after successful addition
                st.session_state.formula_building = ''
                # Note: We do not reset confirmed_formula here due to Streamlit constraints
            else:
                st.error(f"Failed to add parameter: {message}")
 
    if st.sidebar.button('Clear Formula'):
        # Explicit user action to clear the confirmed formula
        st.session_state['confirmed_formula'] = ''


# Function to create plots based on the selected type
def create_plot(plot_type, df, x_axis, y_axis):
    if plot_type == 'Line Plot':
        return px.line(df, x=x_axis, y=y_axis, color='Id', title=f'{y_axis} vs {x_axis}')
    elif plot_type == 'Scatter Plot':
        return px.scatter(df, x=x_axis, y=y_axis, color='Id', title=f'{y_axis} vs {x_axis}')
    elif plot_type == 'Histogram':
        return px.histogram(df, x=y_axis, color='Id', title=f'Histogram of {y_axis}')
    elif plot_type == 'Box Plot':
        return px.box(df, x=y_axis, color='Id', title=f'Box Plot of {y_axis}')
    else:
        return None

# Initialize a session state to manage plots and formula input
if 'plots' not in st.session_state:
    st.session_state['plots'] = []
if 'formula_building' not in st.session_state:
    st.session_state.formula_building = ''

# Main app starts here
st.title('Excel Data Analysis Web App')

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
    
    # Select pairs of columns and plot type
    st.sidebar.header("Analysis Settings")
    columns = df.columns.tolist()
    x_axis = st.sidebar.selectbox('Select X-axis', options=columns, index=columns.index('dayOfExperiment') if 'dayOfExperiment' in columns else 0)
    y_axis = st.sidebar.selectbox('Select Y-axis', options=columns, index=1)
    plot_type = st.sidebar.selectbox('Select Plot Type', options=['Line Plot', 'Scatter Plot', 'Histogram', 'Box Plot'])

    add_custom_parameter_ui(df_filtered)  # UI for adding custom parameters


    # Add plot
    if st.sidebar.button("Add Plot") and len(st.session_state['plots']) < 4:
        fig = create_plot(plot_type, df_filtered, x_axis, y_axis)
        if fig:
            st.session_state['plots'].append(fig)

    # Display plots
    for i, fig in enumerate(st.session_state['plots'], start=1):
        st.plotly_chart(fig, use_container_width=True)
        if st.button(f"Close Plot {i}"):
            st.session_state['plots'].pop(i-1)
            break

    # Correlation Analysis
    st.sidebar.header("Correlation Analysis")
    
    # Let the user select which numeric columns to include in the correlation analysis
    numeric_columns = df_filtered.select_dtypes(include=['number']).columns.tolist()
    selected_columns = st.sidebar.multiselect('Select Columns for Correlation', numeric_columns, default=numeric_columns)
    
    if st.sidebar.button("Show Correlation Matrix"):    
        # Filter the DataFrame based on selected columns
        if selected_columns:
            numeric_df = df_filtered[selected_columns]
            corr_matrix = numeric_df.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", labels=dict(color="Correlation"), title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.sidebar.warning("Please select at least one numeric column for correlation analysis.")

# Running Instructions
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload your Excel file.\n2. Filter subjects if needed.\n3. Select columns for analysis.\n4. Generate plots and view correlation matrix.")
