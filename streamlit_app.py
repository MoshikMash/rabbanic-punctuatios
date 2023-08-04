import streamlit as st
import pickle
import os


def main():
    st.markdown("<h2>Prediction of Rabbinical Punctuation</h2>", unsafe_allow_html=True)
    st.markdown("<u>Description of Predictions:</u>", unsafe_allow_html=True)
    st.markdown("<span style=\"background-color: #ccffcc;\">Green:</span> correct prediction - TP", unsafe_allow_html=True)
    st.markdown("<span style=\"background-color: #FFCCCC;\">Red:</span> Wrong prediction. Square brackets (e.g. [.]) represent the original text's punctuation (if any).", unsafe_allow_html=True)
    st.markdown("[CLS] and [SEP] stand for the beginning and end of each chunk.")
    
    file_list = os.listdir('test_files')
    file_name = st.selectbox('Select File', file_list)

    with open('test_files/' + file_name, 'rb') as file:
        data = pickle.load(file)
    text = data['text']
    html = f'<div style="direction: rtl; text-align: right;">{text}</div>'
    st.markdown(f"<div style='direction: rtl; text-align: right;'>{html}</div></br>", unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()
