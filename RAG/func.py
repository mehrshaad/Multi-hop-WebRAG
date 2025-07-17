from RAG.__import__ import *

load_dotenv()
multiprocessing.freeze_support()
warnings.filterwarnings("ignore")


def create_sbert_mpnet_old(model):
    model = f"sentence-transformers/{model}"
    device = "cpu"
    if torch.cuda.is_available():
        print("Using GPU...", torch.cuda.get_device_name(0))
        device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"Embedding model loaded: {model}")
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"tokenizer": tokenizer},
    )


def create_sbert_mpnet(model):
    model_path = f"sentence-transformers/{model}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={
            "device": device,
            # "torch_dtype": torch.float16 if device == "cuda" else torch.float32
        },
        encode_kwargs={
            "normalize_embeddings": True,  # Critical for ChromaDB
            "batch_size": 64,  # Optimal for most GPUs
            "convert_to_tensor": True,  # Reduces CPU-GPU transfers
            "tokenizer":
            AutoTokenizer.from_pretrained(model_path)  # Moved here
        })


def create_huggingface_client(model='meta-llama/Llama-3.2-3B-Instruct'):
    if model == 'default':
        model = 'meta-llama/Llama-3.2-3B-Instruct'
    client = InferenceClient(api_key=os.getenv("HUGGING_FACE_API_KEY"))

    class OllamaLLM(LLM):

        def _call(self, prompt, stop=None):
            messages = [{"role": "user", "content": prompt}]
            try:
                response = client.chat.completions.create(model=model,
                                                          messages=messages,
                                                          max_tokens=500)

                # Check if response contains valid choices
                if "choices" not in response or not response["choices"]:
                    raise ValueError("Invalid model or empty response.")

                return response["choices"][0]["message"]["content"]
            except Exception as e:
                # Print or log the error
                print(f"Error occurred: {str(e)}")
                return "Error: The model name or API request is invalid."

        @property
        def _llm_type(self):
            return "ollama_llm"

    print(f"Huggingface model loaded: {model}")
    return OllamaLLM(), model


def create_ollama_client(model='llama3.3:70b'):
    if model == 'default':
        model = 'llama3.3:70b'

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    class OllamaLLM(LLM):

        def _call(self, prompt, stop=None):
            url = "https://cosc-llm.cosc.brocku.ca/api/chat/completions"
            headers = {
                "Authorization": f"bearer {os.getenv('OLLAMA_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": 200,
                "temperature": 0.3,
            }

            try:
                resp = requests.post(url,
                                     headers=headers,
                                     json=payload,
                                     verify=False)

                if resp.status_code != 200:
                    raise ValueError(
                        f"API request failed with status code {resp.status_code}"
                    )

                response_data = resp.json()

                # Check if response contains valid choices
                if "choices" not in response_data or not response_data[
                        "choices"]:
                    raise ValueError("Invalid model or empty response.")

                raw_output = response_data["choices"][0].get(
                    "message", {}).get("content", "No response content found.")

                # Post-process to get only the final answer
                match = re.search(r"(?i)^Answer:\s*(.+)$", raw_output.strip(),
                                  re.MULTILINE)
                return match.group(1).strip() if match else raw_output.strip()

            except Exception as e:
                # Print or log the error
                print(f"Error occurred: {str(e)}")
                return "Error: The model name or API request is invalid."

        @property
        def _llm_type(self):
            return "ollama_llm"

    print(f"Ollama model loaded: {model}")
    return OllamaLLM(), model


def create_openai_client(model='gpt-4'):
    if model == 'default':
        model = 'gpt-4'
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    class ChatGPTLLM(LLM):

        def _call(self, prompt, stop=None, model=model):
            messages = [{"role": "user", "content": prompt}]
            try:
                response = client.chat.completions.create(model=model,
                                                          messages=messages,
                                                          max_tokens=500,
                                                          temperature=0.3)

                if not response.choices:
                    raise ValueError("Invalid model or empty response.")

                return response.choices[0].message.content
            except Exception as e:
                # Print or log the error
                print(f"Error occurred: {str(e)}")
                if "429" in str(e):
                    print("Rate limit exceeded, retrying in 30 seconds...")
                    time.sleep(30)
                    return self._call(prompt, stop)
                return "Error: The model name or API request is invalid."

        @property
        def _llm_type(self):
            return "chatgpt_llm"

    print(f"OpenAI model loaded: {model}")
    return ChatGPTLLM(), model


def create_gemini_client(model='gemini-2.0-flash'):
    if model == 'default':
        model = 'gemini-2.0-flash'

    # Import inside function to avoid dependency issues
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    class GeminiLLM(LLM):

        def _call(self, prompt, stop=None):
            time.sleep(2.5)  # Allow time for the API to initialize
            try:
                # Initialize the model with safety settings
                model_obj = genai.GenerativeModel(model_name=model,
                                                  safety_settings={
                                                      'HARASSMENT':
                                                      'block_none',
                                                      'HATE_SPEECH':
                                                      'block_none',
                                                      'SEXUAL': 'block_none',
                                                      'DANGEROUS': 'block_none'
                                                  })

                # Generate content with configurable parameters
                response = model_obj.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=500, temperature=0.3))

                # Handle potential blocking issues
                if response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                    return f"Error: Content blocked ({reason})"

                return response.text.strip()

            except Exception as e:
                print(f"Gemini Error: {str(e)}")
                if "429" in str(e):
                    print("Rate limit exceeded, retrying in 30 seconds...")
                    time.sleep(30)
                    return self._call(prompt, stop)
                return f"Error: {str(e)}"

        @property
        def _llm_type(self):
            return "gemini_llm"

    print(f"Gemini model loaded: {model}")
    return GeminiLLM(), model


def create_chromadb(data_folder, data_name, data_chromadb_output, embedding,
                    chunk_size):
    file_path = f'Data/{data_folder}/{data_name}'

    metadata_path = f'Data/{data_folder}/ChromaDB{data_chromadb_output}/chunk_metadata.json'
    persist_directory = f'Data/{data_folder}/ChromaDB{data_chromadb_output}'

    # Parameters
    chunk_size = chunk_size * 1024 * 1024
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=10)

    # Initialize or load ChromaDB
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)

    # Load progress metadata
    metadata = load_metadata(metadata_path, file_path)
    print(f"Loaded metadata: {metadata}")

    # Open the file and start processing from the last position
    with open(file_path, 'r', encoding='utf-8') as file:
        file.seek(metadata["last_position"])  # Resume from last position
        total_size = len(file.read())  # Get total file size
        file.seek(metadata["last_position"])  # Reset pointer to last position

        # Calculate the total number of chunks
        total_chunks = total_size // chunk_size + (1 if total_size %
                                                   chunk_size > 0 else 0)
        remaining_chunks = (total_size -
                            metadata["last_position"]) // chunk_size + (
                                1 if (total_size - metadata["last_position"]) %
                                chunk_size > 0 else 0)

        print(f"Total chunks to process: {total_chunks}")
        print(f"Resuming with {remaining_chunks} remaining chunks.")

        # Use tqdm to track progress
        with tqdm(total=remaining_chunks, desc="Processing chunks") as pbar:
            while True:
                # Read a chunk of data
                text_data = file.read(chunk_size)
                if not text_data:  # End of file
                    break

                # Split the chunk into smaller text documents
                texts = text_splitter.create_documents([text_data])

                # Add documents to ChromaDB
                vectordb.add_documents(texts)
                vectordb.persist()  # Save database after each chunk

                # Update metadata
                metadata["last_position"] = file.tell()
                metadata["chunk_count"] += 1
                save_metadata(metadata, metadata_path)

                print(f"Processed chunk {metadata['chunk_count']}")
                pbar.update(1)  # Update progress bar

    print(f"Processing complete. Total documents in database: {len(vectordb)}")


def create_chromadb_new(
        data_folder,
        data_name,
        data_chromadb_output,
        embedding,
        chunk_size,
        text_chunk_size=150,  # Updated from 1000 to 150
        persist_interval=50):
    print(f"Generating chromaDB (chunk size: {text_chunk_size})...")
    file_path = f'Data/{data_folder}/{data_name}'
    metadata_path = f'Data/{data_folder}/ChromaDB{data_chromadb_output}/chunk_metadata.json'
    persist_directory = f'Data/{data_folder}/ChromaDB{data_chromadb_output}'

    # Convert MB to bytes for file chunks
    chunk_size_bytes = int(chunk_size * 1024 * 1024)
    chunk_size_bytes = 5461

    # Updated text splitter with chunk_size=150 and overlap=30
    text_splitter = TokenTextSplitter(
        chunk_size=text_chunk_size,
        chunk_overlap=30)  # Explicitly set overlap to 30

    # Rest of the function remains identical...
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)

    metadata = load_metadata(metadata_path, file_path)
    print(f"Metadata loaded: {metadata}")

    total_size = os.path.getsize(file_path)
    current_position = metadata.get("last_position", 0)
    remaining_size = total_size - current_position

    remaining_chunks = remaining_size // chunk_size_bytes + (
        1 if remaining_size % chunk_size_bytes else 0)

    print(f"Remaining chunks: {remaining_chunks}")

    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.seek(current_position)
            with tqdm(total=remaining_chunks,
                      desc="Processing chunks") as pbar:
                processed_chunks = 0

                while True:
                    chunk = mm.read(chunk_size_bytes)
                    if not chunk:
                        break

                    try:
                        text = chunk.decode('utf-8')
                    except UnicodeDecodeError:
                        text = chunk.decode('utf-8', errors='replace')

                    texts = text_splitter.split_text(text)
                    docs = [Document(page_content=t) for t in texts]

                    vectordb.add_documents(docs)
                    processed_chunks += 1

                    current_position += len(chunk)
                    metadata["last_position"] = current_position
                    metadata["chunk_count"] += 1
                    save_metadata(metadata, metadata_path)

                    if processed_chunks % persist_interval == 0:
                        vectordb.persist()

                    pbar.update(1)

                vectordb.persist()

    print(
        f"ChromaDB generation completed. Total documents: {vectordb._collection.count()}"
    )


def load_metadata(metadata_path, file_path):
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"file_path": file_path, "last_position": 0, "chunk_count": 0}


def save_metadata(metadata, metadata_path):
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def get_system_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def get_folder_names(path='Data/'):
    folder_names = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folder_names.append(item)
    return folder_names


def extract_chunks_with_ner(text, nlp):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def split_into_sentences(text, nlp):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def extract_chunks_with_regex(text, pattern):
    matches = re.findall(pattern, text)
    return matches


def extract_chunks_from_json(json_str):
    data = json.loads(json_str)
    chunks = {
        "title": data.get("title"),
        "content": data.get("content"),
        "metadata": data.get("metadata")
    }
    return chunks


def send_email(body: list | str,
               subject='WebRAG Project Progress',
               recipientEmail: list = ['YOURMAIL@gmail.com']):
    if type(body) == str:
        body = [body]
    body = "\n\n".join(body)
    body = [f'Detail: {body}']
    body.append(f'This is an automated message. Do not reply to this message')

    senderEmail = 'YOURMAIL@gmail.com'
    senderKey = 'YOURMAILKEY'
    recipientEmail = recipientEmail
    message = EmailMessage()
    message.set_content("\n\n".join(body))
    message['Subject'] = f"UPDATE: {subject}"
    message['From'] = senderEmail
    message['To'] = ','.join(recipientEmail)
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(senderEmail, senderKey)
        server.send_message(message, senderEmail, recipientEmail)
        server.quit()
        print("Email Sent Successfully!")
    except Exception as e:
        print(f"Error: {str(e).capitalize()}!")


def remove_duplicates(list1, list2):
    set1 = set(list1)
    new_list2 = []
    duplicated_items = []
    for item in list2:
        if item in set1:
            duplicated_items.append(item)
        else:
            new_list2.append(item)
    return duplicated_items, new_list2


def clean_str(s):
    s = s.strip().replace('\n', ' ').replace('\r', ' ').replace(
        '\t',
        ' ').replace('{', '').replace('}', '').replace("\"",
                                                       '').replace('\\',
                                                                   '').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def merge_lists(list1, list2):

    seen = set()
    duplicates = set()
    merged_list = set()

    # First process list1 to populate the seen set and merged_list
    for item in list1:
        item = clean_str(item)
        if item not in seen and len(item) > 3:
            seen.add(item)
            merged_list.add(item)
        elif len(item) > 3:
            duplicates.add(item)

    # Then process list2 to check against seen items
    for item in list2:
        item = clean_str(item)
        if item not in seen and len(item) > 3:
            seen.add(item)
            merged_list.add(item)
        elif len(item) > 3:
            duplicates.add(item)

    return list(merged_list), list(duplicates)
