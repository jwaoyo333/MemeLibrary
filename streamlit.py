import openai
import faiss
import numpy as np
import streamlit as st
import pickle
import json

secrets_file_path = './secret.json'

# 비밀 키 파일에서 API 키 읽기
with open(secrets_file_path) as f:
    secrets = json.loads(f.read())

openai.api_key = secrets["openAi-key"]


# Load precomputed embeddings and FAISS index
with open('faiss_index.pkl', 'rb') as f:
    embeddings_matrix, index, image_filenames = pickle.load(f)

# Function to generate embeddings for user input
def generate_user_input_embedding(user_input):
    response = openai.Embedding.create(
        input=user_input,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def get_relevant_image(user_input):
    user_input_embedding = np.array(generate_user_input_embedding(user_input)).astype('float32').reshape(1, -1)
    _, I = index.search(user_input_embedding, 1)
    most_relevant_image_idx = I[0][0]
    most_relevant_image = image_filenames[most_relevant_image_idx]
    return most_relevant_image

# Streamlit interface
st.title("무도-PT")

user_input = st.text_input("당신의 상태를 말해주세요")

if user_input:
    relevant_image = get_relevant_image(user_input)
    st.image(relevant_image, caption='Relevant Image')
