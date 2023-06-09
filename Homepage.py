import os
import tempfile
import streamlit as st
import openai
from openai.embeddings_utils import get_embedding
import pinecone
import json
import pandas as pd
from transformers import GPT2TokenizerFast
from tqdm.auto import tqdm
import numpy as np
import PyPDF2
from streamlit_chat import message
import streamlit.components.v1 as components
import re
import pyautogui
import atexit


# model
MODEL = "text-embedding-ada-002"

# Create the tokenizer with a maximum sequence length of 2048
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# openai init
@st.cache_resource(show_spinner=False)
def init_openai(openai_key):
    openai.api_key = openai_key

# pinecone init
@st.cache_resource(show_spinner=False)
def load_index(pinecone_key, pinecone_env, index_name):
    pinecone.init(api_key=pinecone_key, environment=pinecone_env)
    index_name = index_name

    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")

    return pinecone.Index(index_name)

# upload file
def read_pdf(filename):
    with open(filename, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
    # Extract the text content from each page
        content = []
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            content.append(page.extract_text())

        # Combine the text content into a single string
        content = '\n'.join(content)

        start = 0
        end = min(len(content), 2000)
        chunk_num = 1
        source_files = []
        while start < len(content):
            chunk_content = content[start:end]
            metadata = f"{filename} (chunk {chunk_num})"
            source_files.append((chunk_content, metadata))
            start = start + 1500
            end = min(len(content), start + 2000)
            chunk_num += 1

        df = pd.DataFrame(source_files, columns=['content','metadata'])
        return df

def process_uploaded_file(uploaded_file, tempName):
    # Read the contents of the uploaded file
    content = uploaded_file.read()

    # Use a temporary file to save the uploaded file contents
    with tempfile.NamedTemporaryFile(delete=False, prefix=tempName) as temp:
        temp.write(content)
        temp.flush()

        # Register a function to delete the temporary file when the script ends
        atexit.register(delete_temp_file, temp.name)
        
        # Call the read_pdf function to extract the text and metadata
        df = read_pdf(temp.name)

    # Combine the text of all chunks into a single string
    #text = ' '.join(df['content'].tolist())

    # Return the processed DataFrame
    return df

def delete_temp_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass

# Define the function to encode text and count tokens
def encode_and_count_tokens(text, tokenizer):
    # Split the text chunk into smaller sub-chunks based on the maximum sequence length
    sub_chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    # Encode each sub-chunk and concatenate the resulting token IDs
    token_ids = []
    for sub_chunk in sub_chunks:
        encoded = tokenizer.encode(sub_chunk, add_special_tokens=False)
        token_ids.extend(encoded)
    # Count the number of tokens
    return len(token_ids)

# Define a function to remove newlines from a text column
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ', regex=False)
    serie = serie.str.replace('\\n', ' ', regex=False)
    serie = serie.str.replace('  ',' ', regex=False)
    serie = serie.str.replace('  ',' ', regex=False)
    return serie

# processing df
def process_df(df, tokenizer, engine):
    df['text'] = "Answer: " + df.content
    df['text'] = remove_newlines(df.text)
    df['n_tokens'] = df['text'].apply(lambda x: encode_and_count_tokens(x, tokenizer))
    df['embeddings'] = df.text.apply(lambda x: get_embedding(x, engine=engine))
    df['id'] = [str(i) for i in range(len(df))]
    return df

# importing to pinecone
def get_metadata_embeddings(df, index, batch_size=32):
    """
    Get the id, metadata, n_tokens and embeddings from the DataFrame `df`
    and upsert them into the `index`.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be upserted.
        index (hnswlib.Index): The HNSW index object to be upserted.
        batch_size (int): The size of each batch of embeddings to be upserted. 
                          Default is 32.
    """
    
    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i+batch_size, len(df))
        df_slice = df.iloc[i:i_end]
        to_upsert = [
            (
                row['id'],
                np.array(row['embeddings']).tolist(),
                {
                    'metadata': row['metadata'],
                    'n_tokens': row['n_tokens']
                }
            ) for _, row in df_slice.iterrows()
        ]
        index.upsert(vectors=to_upsert)


# context generation
def create_context(question, index, mappings, max_len=3750, size="davinci"):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=MODEL)
    res = index.query(q_embed, top_k=3, include_metadata=True)
    
    max_score = max([row['score'] for row in res['matches']])

    cur_len = 0
    contexts = []
    sources = []

    for row in res['matches']:
        if row['id'] in mappings:
            mapped = mappings[row['id']]
            cur_len += row['metadata']['n_tokens'] + 4
            if cur_len < max_len:
                if row['score'] >= max_score * 0.99:
                    contexts.append(mapped)
                    sources.append(row['metadata'])
            else:
                cur_len -= row['metadata']['n_tokens'] + 4
                if max_len - cur_len < 200:
                    break
    sources_list = [source['metadata'] for source in sources]
    return "\n\n###\n\n".join(contexts), sources_list

# answer generation
def answer_question(    
    messages,
    question,
    index,
    mappings,
    fine_tuned_qa_model="text-davinci-002",
    max_len=3550,
    size="davinci",
    debug=False,
    max_tokens=500,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        index,
        mappings,
        max_len=max_len,
        size=size,
    )
    sources = create_context(
                 question,
                 index,
                 mappings,
             )[1]
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        #print(instruction.format(context, question))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
        {"role": "user", "content": "Data: {}\n\n You will answer according to the provided data by remembering the history of the previous text. If the question can't be answered based on the provided data, say \"Sorry, the answer was not found in the document.\"If there is a mathematical operation question then say \"I am not designed to do mathematical operations.\"".format(context)},
        {"role": "assistant", "content": "Okay, got it."},
        {"role": "user", "content": "What is 4*4?"},
        {"role": "assistant", "content": "I am not designed to do mathematical operations."},
        {"role": "user", "content": question}
    ],
            #prompt=instruction.format(context, question),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            #**model_param,
        )
        links = "\n".join([f"{source}" for source in sources])
        return response["choices"][0]["message"]["content"] + "\n\nYou may be interested in visiting: " + links
    except Exception as e:
        print(e)
        return ""


def update_chat(messages, role, content):
  messages.append({"role": role, "content": content})
  return messages

def get_initial_msg():
    messages=[
        {"role": "system", "content": "You are a helpful assistant for chatting with pdf files."}
    ]
    return messages

def get_gpt_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response["choices"][0]["message"]["content"]

# Hiding the default streamlit options
hide_st_default = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_default, unsafe_allow_html=True) 

# Navigation Bar
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-dark bg-dark">
  <div class="container-fluid">
  <a class="navbar-brand">Experimental Chat Application Powered by OpenAI 
  </a>
  </div>
</nav>
 <style>
        .navbar
        {
            border-bottom:3px solid #000;
            position: fixed;
        }
      </style>
""", unsafe_allow_html=True)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


contFileUploader = st.empty()
contPrompt = st.empty()
contSampleQ = st.empty()


def main():
    if 'bot' not in st.session_state:
        st.session_state.bot = []
    if 'user' not in st.session_state:
        st.session_state.user = []
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = ""
    if 'pinecone_key' not in st.session_state:
        st.session_state.pinecone_key = ""
    if 'pinecone_env' not in st.session_state:
        st.session_state.pinecone_env = ""
    if 'index_name' not in st.session_state:
        st.session_state.index_name = "" 

    with st.sidebar:
        st.header("Enter keys for openai & pinecone:old_key:")
        openai_key = st.text_input(label='Enter openai key', type="password")
        pinecone_key = st.text_input(label='Enter pinecone key', type="password")
        pinecone_env = st.text_input(label='Enter pinecone environment')
        index_name = st.text_input(label='Enter index name')
        
        if openai_key and pinecone_key and pinecone_env and index_name:
            st.session_state.openai_key = openai_key
            st.session_state.pinecone_key = pinecone_key
            st.session_state.pinecone_env = pinecone_env
            st.session_state.index_name = index_name

            retriever = init_openai(openai_key=st.session_state.openai_key)
            index = load_index(pinecone_key=st.session_state.pinecone_key, pinecone_env=st.session_state.pinecone_env, index_name=st.session_state.index_name)
        
        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        st.header("Upload files here :page_with_curl:")
        uploaded_files = st.file_uploader("Maximum 50MB", accept_multiple_files=True)

        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        st.header("To start a new chat :left_speech_bubble:")
        if st.button("New Chat"):\
            pyautogui.hotkey("ctrl","F5")

        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        st.header("FAQ :grey_question:")
        st.header("Where do I find openai key?")
        st.write("Please go to https://platform.openai.com/login to sign up for openai key.")
        st.header("Where do I find pinecone keys?")
        st.write("Please go to https://www.pinecone.io/ to sign up for pinecone keys. Use 1536 as dimensions and cosine similarity while setting up the index.")
        st.header("Are the answers 100% accurate?")
        st.write("No, the answers are not 100% accurate. ChatPDFGPT uses GPT-3 to generate answers. It sometimes makes mistakes and is prone to hallucinations. Also, the tool uses semantic search to find the most relevant chunks and does not see the entire document, which means that it may not be able to find all the relevant information and may not be able to answer all questions.")
        st.write("The End")

    with contPrompt.container():
        st.title(":open_book: ChatPDFGPT")
        prompt = st.text_input(label='Enter your question', key='input')

    max_size = 50 * 1024 * 1024
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size > max_size:
                st.error(f"{uploaded_file.name} is larger than 50 MB. Please upload a file smaller than 50 MB.")
            else:
                filename = uploaded_file.name
                if os.path.exists(f'embed/{filename}.json'):
                    with open(f'embed/{filename}.json', 'r') as fp:
                        mappings = json.load(fp)
                else:
                    with st.spinner(f'Processing file {uploaded_file.name}...'):
                        df = process_uploaded_file(uploaded_file = uploaded_file, tempName=uploaded_file.name)
                        df = process_df(df, tokenizer, engine=MODEL)
                        embeddings = get_metadata_embeddings(df, index)
                        mappings_create = {row['id']: row['text'] for _, row in df[['id', 'text']].iterrows()}
                        with open(f'embed/{filename}.json', 'w') as fp:
                            json.dump(mappings_create, fp)

                    with open(f'embed/{filename}.json', 'r') as fp:
                        mappings = json.load(fp)
    else:
        st.warning('Please upload a file.') 
               
    if 'messages' not in st.session_state:
        st.session_state.messages = get_initial_msg()

    if prompt:
        with st.spinner("Generating..."):
            messages = st.session_state.messages
            messages = update_chat(messages, "user", prompt)
            answer = answer_question(
                 messages,
                 prompt,
                 index, 
                 mappings,
             )
            #response = get_gpt_response(messages)
            error_message_pattern = r"Sorry, the answer was not found in the document\.|Therefore, I cannot provide a definitive answer to your question\.|I am not designed to do mathematical operations\.|I'm sorry, I'm not sure what you're asking for\.|Sorry, I'm not sure what you're asking for."
            error_match = re.search(error_message_pattern, answer)
            if error_match:
                response = "Sorry, the answer was not found in the uploaded pdf file."
            else:
                response = answer
            messages = update_chat(messages, "assistant", response)
            messages = update_chat(messages, "user", prompt)
            st.session_state.user.append(prompt)
            st.session_state.bot.append(response)

            # Scroll down to the bottom of the page
            scroll_js = "window.scrollTo(0, document.body.scrollHeight);"
            components.html('<script>{}</script>'.format(scroll_js))
    
    with st.container():
        if st.session_state.bot:
            bot = st.session_state.bot
            user = st.session_state.user
            for i in range(len(bot)):
            #for i in range(len(st.session_state.bot)-1,-1,-1):
                message(user[i], is_user=True, key=str(i) + '_user')
                message(bot[i], key=str(i))

if __name__=="__main__":
    main()