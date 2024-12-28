import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download the required NLTK resources
try:
    nltk.download('punkt_tab')  # Download the correct punkt tokenizer
    nltk.download('stopwords')  # Download the stopwords data
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    raise

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform the text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text using word_tokenize
    try:
        text = word_tokenize(text)
    except Exception as e:
        st.error(f"Error in tokenizing text: {e}")
        raise

    # Remove non-alphanumeric tokens (numbers, punctuation)
    text = [i for i in text if i.isalnum()]

    # Remove stopwords
    text = [i for i in text if i not in stopwords.words('english')]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load the pretrained models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files (vectorizer.pkl or model.pkl) not found. Please make sure they are in the correct path.")
    raise

# Streamlit app
st.title("Email/SMS Spam Classifier")

# Input text area
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the preprocessed text using the trained vectorizer
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the pretrained model
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to classify.")
