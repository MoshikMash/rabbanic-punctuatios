import streamlit as st
import pickle
import os


def main():
    st.markdown("<span style=\"background-color: #ccffcc;\">Green:</span> correct prediction – TP", unsafe_allow_html=True)
    st.markdown("<span style=\"background-color: #FFCCCC;\">Red:</span> Wrong prediction. Wherever there are square brackets (e.g. [.]), these square brackets represent the punctuation in the text, and the punctuation without brackets represents the prediction. If there is only punctuation within brackets and no punctuation without the brackets, it means that the prediction was BLANK (no punctuation), or, alternately, if there is no punctuation with brackets, and only punctuation without brackets, it means that there was no punctuation in the original text (only in the prediction). For example:", unsafe_allow_html=True)
    st.markdown("[CLS] and [SEP] the beginning and the end of each chunk.")
    
    file_list = os.listdir('test_files')
    file_name = st.selectbox('Select File', file_list)

    with open('test_files/' + file_name, 'rb') as file:
        data = pickle.load(file)
    text = data['text']
    html = f'<div style="direction: rtl; text-align: right;">{text}</div>'
    st.markdown(f"<div style='direction: rtl; text-align: right;'>{html}</div></br>", unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()
