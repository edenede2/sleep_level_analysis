import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        return False, error_message, None

    try:
        # Assuming df.eval can handle the formula after type checks and conversions
        df[param_name] = df.eval(formula)
        return True, "Parameter added successfully!", df
    except Exception as e:
        return False, str(e), None

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
            # Reset the trigger to default to ensure consistency
            st.session_state.reset_formula_flag = False

    # User input for final formula modification or confirmation
    # Use the reset trigger to conditionally set the default value
    default_formula_value = '' if st.session_state.reset_formula_flag else st.session_state.formula_building
    confirmed_formula = st.sidebar.text_input('Confirm or modify formula', value=default_formula_value, key='confirmed_formula')
    
    # Button to add the parameter
    if st.sidebar.button('Add Parameter'):
        if param_name and confirmed_formula:
            success, message, updated_df = evaluate_formula(df, confirmed_formula, param_name)
            if success:
                st.success(message)
                # Reset formula building state after successful addition
                st.session_state.formula_building = ''
                st.session_state['df_modified'] = updated_df
                # Note: We do not reset confirmed_formula here due to Streamlit constraints
            else:
                st.session_state.formula_building = ''
                st.error(f"Failed to add parameter: {message}")
 
    if st.sidebar.button('Clear Formula'):
        # Explicit user action to clear the confirmed formula
        st.session_state.reset_formula_flag = not st.session_state.reset_formula_flag
        st.session_state.formula_building = ''

        



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
if 'reset_formula_flag' not in st.session_state:
    st.session_state.reset_formula_flag = False

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

    st.session_state['df_modified'] = df.copy()


    # Showing the dataframe
    st.write("Data Preview:")
    st.write(df.head())
    
        # Data Concatenation Settings Section
    st.sidebar.header("Data Concatenation Settings")
    concat_criteria = st.sidebar.selectbox('Select Criterion for Concatenation', ['None'] + list(df.columns), index=0)
    concat_value = None
    if concat_criteria != 'None':
        concat_value = st.sidebar.text_input(f"Enter value for {concat_criteria} to filter and concatenate (e.g., '1' for Day 1):", key='concat_value')

    concatenated_df = pd.DataFrame()

    if concat_criteria != 'None' and concat_value:
        try:
            # Convert concat_value to appropriate type, assuming it's numeric for simplicity
            concat_value = int(concat_value)
            filtered_df = df[df[concat_criteria] == concat_value]
        except ValueError:
            st.error("Please enter a valid numeric value for concatenation.")
            filtered_df = pd.DataFrame()

        if not filtered_df.empty:
            concatenated_df = pd.concat([concatenated_df, filtered_df])
            st.write("Concatenated Data Preview:")
            st.write(concatenated_df.head())
        else:
            st.warning("No data matches the specified criteria for concatenation.")
    else:
        concatenated_df = df.copy()  # Use the original DataFrame if no concatenation criteria are specified

    # Filtering subjects from the concatenated or original DataFrame
    st.sidebar.header("Filter Subjects")
    unique_subjects = concatenated_df['Id'].unique()
    subjects_to_analyze = st.sidebar.multiselect('Select Subjects', unique_subjects, default=unique_subjects)
    
    # Filtering concatenated dataframe based on selected subjects
    df_filtered = concatenated_df[concatenated_df['Id'].isin(subjects_to_analyze)]

    # Custom parameter UI and analysis setup go here
    add_custom_parameter_ui(df_filtered)

    # Select pairs of columns and plot type
    st.sidebar.header("Analysis Settings")
    df_filtered = st.session_state['df_modified'][st.session_state['df_modified']['Id'].isin(subjects_to_analyze)]

    columns = df_filtered.columns.tolist()
    x_axis = st.sidebar.selectbox('Select X-axis', options=columns, index=columns.index('dayOfExperiment') if 'dayOfExperiment' in columns else 0)
    y_axis = st.sidebar.selectbox('Select Y-axis', options=columns, index=1)
    plot_type = st.sidebar.selectbox('Select Plot Type', options=['Line Plot', 'Scatter Plot', 'Histogram', 'Box Plot'])

    # Plotting based on user selections
    if st.sidebar.button("Add Plot") and len(st.session_state['plots']) < 4:
        fig = create_plot(plot_type, df_filtered, x_axis, y_axis)
        if fig:
            st.session_state['plots'].append(fig)

    for i, fig in enumerate(st.session_state['plots'], start=1):
        st.plotly_chart(fig, use_container_width=True)
        if st.button(f"Close Plot {i}"):
            st.session_state['plots'].pop(i-1)
            break

    # Correlation Analysis
    st.sidebar.header("Correlation Analysis")
    numeric_columns = df_filtered.select_dtypes(include=['number']).columns.tolist()
    selected_columns = st.sidebar.multiselect('Select Columns for Correlation', numeric_columns, default=numeric_columns)

    color_scale = [
            [-1, "red"],  # Negative correlation
            [0, "white"],  # No correlation
            [1, "blue"]  # Positive correlation
        ]

    
    if st.sidebar.button("Show Correlation Matrix"):    
        if selected_columns:
            numeric_df = df_filtered[selected_columns]
            corr_matrix = numeric_df.corr()
        
            # Prepare hover text that includes the names of the parameters and their correlation value
            hover_text = [[f"{row}, {col}<br>Correlation: {corr_matrix.loc[row, col]:.2f}" for col in corr_matrix.columns] for row in corr_matrix.index]
        
            # Define colors based on the correlation values: blue for positive, red for negative
            colors = np.where(corr_matrix.values > 0, 'blue', 'red')  # Blue for positive, red for negative correlations
            colors[corr_matrix.values == 0] = 'white'  # Optional: set exactly 0 correlations to white
        
            # Create the figure with Graph Objects
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                text=np.round(corr_matrix, 2),  # Optionally round the correlation values for display
                texttemplate="%{text}",
                hoverongaps=False,
                hoverinfo="text",
                colorscale=[[0, 'red'], [0.5, 'white'], [1, 'blue']],  # This is not directly used but kept for reference
                showscale=False,  # Optionally turn off the color scale since we're manually setting colors
                colorbar=dict(title='Correlation'),
            ))
        
            # Update layout to add titles and adjust as necessary
            fig.update_layout(title_text='Correlation Matrix', title_x=0.5)
        
            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.sidebar.warning("Please select at least one numeric column for correlation analysis.")

# Running Instructions
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload your Excel file.\n2. Filter subjects if needed.\n3. Select columns for analysis.\n4. Generate plots and view correlation matrix.")
