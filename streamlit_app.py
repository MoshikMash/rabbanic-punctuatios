import streamlit as st
import pandas as pd
import re

if 'res_df' not in st.session_state:
    st.session_state.res_df = pd.read_pickle('res_df.p')
if 'number_filename_dict' not in st.session_state:
    st.session_state.number_filename_dict = pd.read_pickle('number_filename_dict.p')

# Get unique filenames
unique_filenames = st.session_state.res_df['file_name'].unique()

# Function to apply colors to the text based on HTML styling


# Streamlit app
st.title("Text Comparison App")

st.write("""
       Notes: 
       1. The interface examines the first three chunks of the selected text.
       2. Green highlights represent the same punctuation marks as in the original text.
       3. Red highlights represent differences between the predicted punctuation marks and the punctuation marks in the original text. 
       """)

# Select box for choosing a filename
selected_filename = st.selectbox("Select Filename", unique_filenames)
file_num = st.session_state.number_filename_dict[selected_filename]


# Display the text side by side with colors
selected_data = st.session_state.res_df[st.session_state.res_df['unique_number'] == file_num]
if not selected_data.empty:
    original_text = []
    processed_text = []
    for i in range(3):
        gt_text = selected_data['gt'].iloc[0][i]
        pred_text = selected_data['pred'].iloc[0][i]
        original_text.append(gt_text)
        processed_text.append(pred_text)


    processed_df = pd.DataFrame({'טקסט מקור': original_text, 'טקסט מתוקן': processed_text})

    table_html = processed_df.to_html(index=False, escape=False)

    table_html = table_html.replace('<td>', '<td><div class="cell" id="{{col}}-{{row}}">')
    table_html = table_html.replace('</td>', '</div></td>')
    table_html = table_html.replace('<table', '<table style="direction: rtl;"')

    st.write(table_html, unsafe_allow_html=True)

    # Create columns for side-by-side display
    # col1, col2 = st.columns(2)

    # # Display Ground Truth in the first column
    # with col1:
    #     rtl_text = "מודל ניבוי"
    #     styled_rtl_text = f'<div dir="rtl" style="text-align:center; font-weight:bold;">{rtl_text}</div>'
    #     st.markdown(styled_rtl_text, unsafe_allow_html=True)
    # with col2:
    #     rtl_text = "טקסט מקור"
    #     styled_rtl_text = f'<div dir="rtl" style="text-align:center; font-weight:bold;">{rtl_text}</div>'
    #     st.markdown(styled_rtl_text, unsafe_allow_html=True)


else:
    st.warning("No data found for the selected filename.")
