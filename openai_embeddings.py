from openai import OpenAI
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv

'''# Load environment variables from .env file
load_dotenv()
# Access the API key from the environment
OpenAI.api_key = os.getenv('OPENAI_API_KEY')'''


# Create a DataFrame with the combined data

# If using google CoLab:
# from google.colab import drive
# drive.mount('/content/drive')

# If local storage:
current_dir = os.getcwd()
sample_data_directory = os.path.join(current_dir, "sampledata/precompressed/llm_precompressed")

#Generate the data frame
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

# Function to process a single file and return its data
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    segments = re.split(r'\[LLM GENERATED SUMMARY\]', content)
    data = []
    for i in range(1, len(segments), 2):
        summary_segment = segments[i].strip()
        if i - 1 >= 0:
            previous_text = segments[i - 1].strip()
            timestamp_match = re.search(r'\[timestamp: (.*?)\]', previous_text)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                previous_text = re.sub(r'\[timestamp: .*?\]', '', previous_text).strip()
            else:
                timestamp = None
            data.append([summary_segment, previous_text, timestamp])
    return data

# Initialize an empty list to store all data
all_data = []

# Loop through all text files in the directory and process them
for root, dirs, files in os.walk(sample_data_directory):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            file_data = process_file(file_path)
            all_data.extend(file_data)

# Create a DataFrame with the combined data
df = pd.DataFrame(all_data, columns=['Summary', 'Text', 'Timestamp'])

# Extract summaries into a list
# IMPORTANT FOR SUBSEQUENT CODE FLOW
summaries_list = df['Summary'].tolist()

## String search: this code is independent of LLM providers
# Define the search function
def find_summaries(search_string, summaries):
    locations = [index for index, summary in enumerate(summaries) if search_string.lower() in summary.lower()]
    return locations

# Example usage
search_string = "some search string"
locations = find_summaries(search_string, summaries_list)
print(f"Summaries containing '{search_string}' are found at indices: {locations}")

# Print the actual summaries found for verification
for loc in locations:
    print(f"\nSummary at index {loc}:\n{summaries_list[loc]}")

## Embedding code for OpenAI:
openai_model_name = 'text-search-ada-query-001'

# Function to get embeddings for a list of texts in batches
def get_embeddings(texts, client, model=openai_model_name, batch_size=20):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(input=batch, model=model)
            # Adding detailed debug information
            #print(f"Response: {response}")
            for res in response.data:
                #print(f"Embedding: {res.embedding}")
                embeddings.append(res.embedding)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
    embeddings = np.array(embeddings)
    #print(f"Embeddings shape: {embeddings.shape}")  # Check the shape of embeddings
    return embeddings

# Function to prepare and index embeddings for summaries
def prepare_index(summaries, client, model=openai_model_name, batch_size=20):
    embeddings = get_embeddings(summaries, client, model, batch_size)
    if len(embeddings) == 0:
        raise ValueError("No embeddings were generated.")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings

# Function to retrieve relevant summaries based on a query
def retrieve_summaries(query, summaries, client, model=openai_model_name, threshold=0.75, batch_size=20):
    index, embeddings = prepare_index(summaries, client, model, batch_size)
    query_embedding = get_embeddings([query], client, model, batch_size=batch_size).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding, len(summaries))
    similarities = 1 - (D / np.max(D))
    relevant_summaries = [summaries[i] for i, similarity in zip(I[0], similarities[0]) if similarity > threshold]
    relevant_similarities = [similarity for similarity in similarities[0] if similarity > threshold]
    return relevant_summaries, relevant_similarities

# Testing with a single summary
single_summary_list = summaries_list[:1]
query = "enter some search query here"
threshold = 0.3  # THIS IS A CRITICAL VARIABLE
results = retrieve_summaries(query, single_summary_list, client, model="text-embedding-3-small", threshold=threshold)
relevant_summaries, relevant_similarities = results
#print("Relevant summaries:", relevant_summaries)
#print("Relevant similarities:", relevant_similarities)

# Print the results
for idx, (summary, similarity) in enumerate(zip(relevant_summaries, relevant_similarities)):
    print(f"Summary {idx+1} (Similarity: {similarity:.2f}):\n{summary}\n")
