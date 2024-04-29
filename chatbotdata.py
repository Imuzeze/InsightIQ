import nltk
import random 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
#import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

#Download required NLTK data 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


data = pd.read_csv('chatbot dataset.txt', sep = "\t", header = None)
data.rename(columns = {0: 'Question', 1: 'Answer'}, inplace = True)


              
# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['Question'].apply(preprocess_text)


xtrain = data['tokenized Questions'].to_list()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)


#---------------------Streamlit Implementation---------------------

st.markdown(
    """
    <style>
    body {
        #D9CAFAA
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Define your description
desc = "Personalized Chatbot for Curated Knowledge Discovery"


st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>INSIGHTIQ CHATBOT</h1>", unsafe_allow_html = True)

# Create a scrolling marquee for description
desc_text = "Personalized Chatbot for Curated Knowledge Discovery"
marquee_html = f"<marquee style='font-size: 20px; color: #132043; font-family: montserrat;' scrollamount='5'>{desc_text}</marquee>"
st.markdown(marquee_html, unsafe_allow_html=True) 

# st.markdown("<h4 style = 'margin: -30px; color: #803D3B; text-align: center; font-family: cursive '>Personalized Chatbot for Curated Knowledge Discovery</h4>", unsafe_allow_html = True)
# st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'margin: -30px; color: #948979; text-align: center; font-family: cursive; font-size: smaller; '>Built By Joshua Uanikehi</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

#To add anything to the side bar of the page
st.sidebar.image('pngwing.com (5).png')

#To add space to the sidebar before adding writeup to give line spaace
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

#Quick Links
st.sidebar.markdown("<h3 style='color: #0C2D57; font-family: geneva;'>Explore with INSIGHTIQ</h3>", unsafe_allow_html=True)
# st.sidebar.markdown("<button style='background-color: #0C2D57; color: white; padding: 10px 20px; border: none; border-radius: 5px;'>About InsightIQ</button>", unsafe_allow_html=True)
# st.sidebar.markdown("<button style='background-color: #0C2D57; color: white; padding: 10px 20px; border: none; border-radius: 5px;'>FAQs</button>", unsafe_allow_html=True)
# st.sidebar.markdown("<button style='background-color: #0C2D57; color: white; padding: 10px 20px; border: none; border-radius: 5px;'>Start Chat</button>", unsafe_allow_html=True)

# Define functions to display additional information
def display_AboutInsightIQ():
    st.write("""InsightIQ is an AI-powered chatbot designed to provide quick and accurate answers to a wide range of questions across various topics. Leveraging advanced natural language processing (NLP) techniques, InsightIQ serves as an interactive tool for users seeking personalized assistance and curated knowledge discovery.""")

    # Key Features:
    # Wide Topic Coverage: InsightIQ is equipped to handle inquiries on diverse topics, ensuring comprehensive knowledge discovery.
    # Quick and Accurate Responses: With advanced NLP techniques, InsightIQ delivers quick and accurate responses tailored to users queries.
    # Interactive Experience: InsightIQ offers an engaging and interactive experience, allowing users to explore new topics and find relevant information effortlessly.

    # Whether you are looking for recommendations, exploring new interests, or seeking quick answers, InsightIQ is your go-to chatbot for personalized assistance and curated knowledge discovery.
 # Add Get Started Guide content here   

def display_FAQs():
    st.write("""" 1. What is InsightIQ?  An AI-powered chatbot providing quick and accurate answers across various topics.\n ***2. How does it work?  Utilizes advanced NLP techniques to understand queries and deliver personalized responses.\n ***3. What topics does it cover?  Diverse topics for comprehensive knowledge discovery.\n 4. How accurate are the responses?
    Quick and accurate, thanks to advanced NLP capabilities.\n ***5. Can I interact with it?
    Yes, it offers an engaging and interactive experience.\n ***6. How to provide feedback or report issues?
    Contact our support team through provided channels.\n ***7. Is it free to use?
    Yes, it's free for all users seeking quick answers. """)
# Add FAQ content here


def display_StartChat():
    st.write("""
    **Ready to discover insights and get quick answers?**

    Start chatting with InsightIQ, your AI-powered assistant, to explore new topics, find recommendations, and get personalized assistance.""")
    # Add Feedback Form content here

# Create expanders for each quick link
with st.sidebar.expander("About InsightIQ"):
    display_AboutInsightIQ()

with st.sidebar.expander("FAQs"):
    display_FAQs()

with st.sidebar.expander("Start Chat"):
    display_StartChat()







user_hist = []
reply_hist = []


robot_image, space1, space2, chats = st.columns(4)
with robot_image: 
    robot_image.image('pngwing.com (2).png', width = 400)

with chats:
    user_message = chats.text_input('Hello there you can ask your questions: ')
    def responder(user_input):
        user_input_processed = preprocess_text(user_input)
        vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
        similarity_score = cosine_similarity(vectorized_user_input, corpus)
        argument_maximum = similarity_score.argmax()
        print (data['Answer'].iloc[ argument_maximum])

bot_greetings = ['Hello user, i am a creation of zeze the great...Ask your question',
             'How far wetin dey sup?',
             'How may i help you?',
             'Why you show face, everything clear?',
             'Good day user, welcome to my world. How may i help you?']

bot_farewell = [ 'Thanks for your usage... bye',
            'Alright sir... Hope to see you soon',
            'Oya now... e go be',
            'Everygood abi.. later things']

human_greetings = ['hi', 'hello there', 'hey', 'hello']

human_exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']

import random
random_greeting = random.choice(bot_greetings)
random_farewell = random.choice(bot_farewell)


# Clearing Chat History 
def clearHistory():
    with open('history.txt', 'w') as file:
        pass  

    with open('reply.txt', 'w') as file:
        pass

if user_message.lower() in human_exits:
    chats.write(f"\nChatbot: {random_farewell}!")
    user_hist.append(user_message)
    reply_hist.append(random_farewell)

elif user_message.lower() in human_greetings:
    chats.write(f"\nChatbot: {random_greeting}!")
    user_hist.append(user_message)
    reply_hist.append(random_greeting)

elif user_message == '':
    chats.write('')

else:
    response = responder(user_message)
    chats.write(f"\nChatbot: {response}!")
    user_hist.append(user_message)
    reply_hist.append(response)


#save the history of user texts
import csv
with open('history.txt', 'a') as file:
    for item in user_hist:
        file.write(str(item) + '\n')

#save history of bot reply
with open('reply.txt', 'a') as file:
    for item in reply_hist:
        file.write(str(item) + '\n')

#import the file to display it in the frontend 
with open('history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('reply.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({'User Input': data1, 'Bot Reply': data2})

#history = pdSeries(data)
st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)
#st.sidebar.write(data2)

#Button For Chat History Clearing
if st.button('Clear Chat History'):
    clearHistory()

st.markdown("<br>", unsafe_allow_html= True)

# Feedback and Reviews
st.subheader('Feedback and Reviews', divider=True)

# Flag to control the visibility of the feedback form
show_feedback_form = st.checkbox("Leave Feedback")

# Display feedback form if checkbox is checked
if show_feedback_form:
    feedback = st.text_area("Leave your feedback or review here:")
    submit_button = st.button("Submit")

    if submit_button and feedback:
        # You can store the feedback in a database or simply display it
        st.success("Thank you for your feedback! Your review has been submitted.")
        # Clear the feedback form
        show_feedback_form = ""


st.markdown("<br>", unsafe_allow_html= True)


# Contact Information
st.subheader('Contact Information', divider=True)
st.markdown("You're inspired to build your own bot? Let's chat!")
st.markdown("- Phone: +2348130957929")
st.markdown("- Instagram: [Instagram](https://www.instagram.com/glotexpaints?igsh=OGQ5ZDc2ODk2ZA==)")
st.markdown("- Email: uanikehijoshua@gmail.com")
st.markdown("- Address: 12, Connal Road, Yaba, Lagos")
