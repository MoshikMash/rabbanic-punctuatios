import streamlit as st
import pickle
import os


def main():
    st.markdown("<u>Prediction Description</u>", unsafe_allow_html=True)
    st.markdown("<span style=\"background-color: #ccffcc;\">Green:</span> correct prediction – TP", unsafe_allow_html=True)
    st.markdown("<span style=\"background-color: #FFCCCC;\">Red:</span> Wrong prediction. square brackets (e.g. [.]) represent the punctuation in the original text (if any).", unsafe_allow_html=True)
    st.markdown("[CLS] and [SEP] the beginning and the end of each chunk.")
    
    file_list = os.listdir('test_files')
    file_name = st.selectbox('Select File to Predict Punctuations', file_list)

    with open('test_files/' + file_name, 'rb') as file:
        data = pickle.load(file)
    text = data['text']
    html = f'<div style="direction: rtl; text-align: right;">{text}</div>'
    st.markdown(f"<div style='direction: rtl; text-align: right;'>{html}</div></br>", unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()
