from RAG.__import__ import *
from RAG.func import *
from RAG.relatedData import load_wiki_data, extract_related_data, get_related_sentences_hotpot

DATA_MAP = {
    '2WikiMultihopQA': {
        'name': 'wiki_data',
        'extentions': [
            '-full-list',
            '-demo',
            '-full-dict',
        ],
        'relationships': '',
        'questions_diversity': {
            "bridge_comparison": 0.21,
            "compositional": 0.46,
            "inference": 0.02,
            "comparison": 0.31
        }
    },
    'HotpotQA': {
        'name': 'hotpot',
        'extentions': [
            '-dev-context',
        ],
        'relationships': '',
        'questions_diversity': {
            "comparison": 0.20,
            "bridge": 0.80
        }
    },
    'CloudComputing': {
        'name': 'cloud_data',
        'extentions': ['_LDA', '_LLM', '_RAW'],
        'relationships': '-link-new'
    }
}


class RAG:

    def __init__(self,
                 data_folder: Literal['2WikiMultihopQA', 'HotpotQA',
                                      'CloudComputing'],
                 chromadb_extension_name: str,
                 mode: Literal['rag', 'rag+related_data', 'related_data',
                               'rag&related_data'],
                 embedding_model: Literal["all-mpnet-base-v2",
                                          "all-MiniLM-L6-v2"],
                 llm_provider: Literal['openai', 'huggingface', 'ollama',
                                       'gemini'],
                 llm_model: str = 'default',
                 additional_results_name: str = ''):

        self.embedding = create_sbert_mpnet(model=embedding_model)
        self.mode = '-' + mode
        # self.mode = '-relationships' if relationships else ''
        self.relationships_context = ''
        self.llm_provider = llm_provider
        if llm_provider.lower() == 'openai':
            self.llm, self.llm_model = create_openai_client(model=llm_model)
            # embedding = OpenAIEmbeddings(show_progress_bar=True, chunk_size=5)
        elif llm_provider.lower() == 'huggingface':
            self.llm, self.llm_model = create_huggingface_client(
                model=llm_model)
        elif llm_provider.lower() == 'ollama':
            self.llm, self.llm_model = create_ollama_client(model=llm_model)
        elif llm_provider.lower() == 'gemini':
            self.llm, self.llm_model = create_gemini_client(model=llm_model)

        # Input Paths
        self.data_folder = data_folder
        self.additional_results_name = additional_results_name

        if chromadb_extension_name in DATA_MAP[data_folder]['extentions']:
            self.data_chromadb_output = chromadb_extension_name
        else:
            raise ValueError(
                f"Invalid extension name: {chromadb_extension_name}. Valid extensions are: {DATA_MAP[data_folder]['extentions']}"
            )

        self.data_name = f'{DATA_MAP[data_folder]["name"]}{chromadb_extension_name}.json'

        # Paths
        self.persist_directory = f'./Data/{self.data_folder}/ChromaDB{self.data_chromadb_output}'
        self.vectordb = False

    def create_chromadb(self,
                        chunk_size: int | float = 2,
                        use_new: bool = True):
        if use_new:
            create_chromadb_new(self.data_folder, self.data_name,
                                self.data_chromadb_output, self.embedding,
                                chunk_size)
        else:
            create_chromadb(self.data_folder, self.data_name,
                            self.data_chromadb_output, self.embedding,
                            chunk_size)

    def get_related_data(self, question: str):
        # Load data once (will be cached)
        if self.data_folder == '2WikiMultihopQA':
            data_cache = load_wiki_data()

            # Get sentences using cached data
            sentences = extract_related_data(question, data_cache)

            if sentences:
                self.related_data = [
                    clean_str(sentence) for sublist in sentences
                    for sentence in sublist if len(sentence) > 1
                ]
            else:
                print("No related data found for the question. Using default.")
                self.related_data = []
        elif self.data_folder == 'HotpotQA':
            self.related_data = get_related_sentences_hotpot(
                question_text=question)
            self.related_data = [
                clean_str(sentence) for sentence in self.related_data
                if len(sentence) > 1
            ]

    def load_questions(self, fname: str = 'questions'):
        print('Loading questions...')
        path = f'./Data/{self.data_folder}/Questions/{fname}.json'

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions from {path}")
        questions = pd.DataFrame(data)

        return questions

    def load_relationships(self):
        if 'related_data' not in self.mode: return
        path = f'./Data/{self.data_folder}/{DATA_MAP[self.data_folder]["name"]}{self.data_chromadb_output}{DATA_MAP[self.data_folder]["relationships"]}.json'
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.data_folder == '2WikiMultihopQA':
            pass
        elif self.data_folder == 'CloudComputing':
            relationships = []
            for vendor, vendor_items in data.items():
                for product in vendor_items:
                    if 'connections' not in product.keys():
                        continue
                    text = f'{product["name"]} is connected to ' + ' and '.join(
                        product['connections'])
                    relationships.append(text)
        self.relationships_context = '\n'.join(relationships)
        print(f'Loaded relationships from {path}')

    def load_system_prompt(self):
        path = f'prompt-{self.data_folder}'
        # self.load_relationships()

        with open(f'SystemPrompts/{path}.txt', 'r') as file:
            data = file.read()

        data = data.replace('RELATIONSHIPS', self.relationships_context)
        self.question_template = data
        print(f"Loaded system prompt from SystemPrompts/{path}.txt")

    def load_results(self, fname: str = 'results'):
        try:
            if 'rag' in self.mode:
                path = f'./Data/{self.data_folder}/Results/{self.additional_results_name}{fname}{self.data_chromadb_output}{self.mode}-{self.llm_provider}-k{self.k}.json'
            else:
                path = f'./Data/{self.data_folder}/Results/{self.additional_results_name}{fname}{self.data_chromadb_output}{self.mode}-{self.llm_provider}.json'
            with open(path, 'r', encoding='utf-8') as results_file:
                data = json.load(results_file)
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} previous results from {path}")
            return df
        except FileNotFoundError:
            print(f"No results file found in {path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()

    def save_results(self, results: pd.DataFrame, fname: str = 'results'):
        if 'rag' in self.mode:
            path = f'./Data/{self.data_folder}/Results/{self.additional_results_name}{fname}{self.data_chromadb_output}{self.mode}-{self.llm_provider}-k{self.k}.json'
        else:
            path = f'./Data/{self.data_folder}/Results/{self.additional_results_name}{fname}{self.data_chromadb_output}{self.mode}-{self.llm_provider}.json'
        directory = os.path.dirname(path)
        try:
            os.makedirs(directory, exist_ok=True)
            results.to_json(path,
                            orient='records',
                            force_ascii=False,
                            indent=1)
            print(f"Saved {len(results)} results to {path}")
        except Exception as e:
            print(f"An error occurred while saving results: {e}")

    def start(self, k: int):
        self.vectordb = Chroma(persist_directory=self.persist_directory,
                               embedding_function=self.embedding)
        self.num_documents = self.vectordb._collection.count()
        print(
            f"ChromaDB loaded. Number of documents in the database: {self.num_documents}"
        )
        self.k = k
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm,
                                              chain_type="stuff",
                                              retriever=self.retriever)
        self.load_system_prompt()
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def run(self, question: str):
        self.related_data = []
        if 'related_data' in self.mode:
            self.get_related_data(question)

        if 'rag---' in self.mode:
            prompt = PromptTemplate(
                template=self.question_template.replace(
                    "RELATED_DATA", ", ".join(self.related_data)),
                input_variables=["context", "question"],
            )
            self.qa.combine_documents_chain.llm_chain.prompt = prompt
            ret = self.get_context(question=question, document=True)
            response = self.qa({
                "query": question,
                "context": ret,
            })

            return response

        elif 'rag&related_data' in self.mode:
            self.QUESTION_PROMPT = PromptTemplate(
                template=self.question_template.replace(
                    "{context}", ', '.join(
                        self.get_context(question=question,
                                         return_duplicates=True))),
                input_variables=["question"])
            self.qa.combine_documents_chain.llm_chain.prompt = self.QUESTION_PROMPT
            response = self.qa({
                "query": question,
            })
            return response

        elif 'rag' in self.mode:
            self.QUESTION_PROMPT = PromptTemplate(
                template=self.question_template.replace(
                    "{context}", ', '.join(
                        self.get_context(question=question,
                                         return_duplicates=False))),
                input_variables=["question"])
            self.qa.combine_documents_chain.llm_chain.prompt = self.QUESTION_PROMPT
            response = self.qa({
                "query": question,
            })
            return response

        elif 'related_data' in self.mode:
            self.QUESTION_PROMPT = PromptTemplate(
                template=self.question_template.replace(
                    '{context}', ', '.join(self.related_data)),
                input_variables=["question"])
            self.qa.combine_documents_chain.llm_chain.prompt = self.QUESTION_PROMPT
            response = self.qa({
                "query": question,
            })
            return response

    def get_context(self,
                    question: str,
                    document: bool = False,
                    k: int = 0,
                    return_duplicates: bool = False,
                    delete_duplicates: bool = True):
        if self.mode == '-related_data':
            return self.related_data
        ret = self.retriever.invoke(question)
        if k:
            ret = self.vectordb.as_retriever(search_kwargs={
                "k": k
            }).invoke(question)

        if document: return ret

        context = []

        for doc in ret:
            # context.append(str(list(doc)[2][-1]).strip())
            docs = str(list(doc)[2][-1]).strip()
            docs = [clean_str(item) for item in docs.split(',\n  "')]
            context += docs

        if 'related_data' in self.mode and (delete_duplicates
                                            or return_duplicates):
            self.get_related_data(question)
            context, self.duplicates = merge_lists(self.related_data, context)
            if return_duplicates:
                print(
                    f"Found {len(self.duplicates)} duplicates in related data."
                )
                return self.duplicates
        return list(context)

    def get_context_with_custom_extraction(
        self,
        question: str,
        method: Literal['NER', 'regex', None],
        k: int = 0,
        pattern:
        str = r"Date: (\d{4}-\d{2}-\d{2})|Amount: (\$\d+)|Description: ([A-Za-z\s]+)",
        nlp_model: Literal["en_core_web_sm", 'en_core_web_md',
                           "en_core_web_lg",
                           "en_core_web_trf"] = 'en_core_web_md'):

        nlp = spacy.load(nlp_model)
        ret = self.retriever.invoke(question)
        if k:
            ret = self.vectordb.as_retriever(search_kwargs={
                "k": k
            }).invoke(question)

        context = []
        for doc in ret:
            text = str(list(doc)[2][-1]).strip()
            if method == "NER":
                chunks = extract_chunks_with_ner(text, nlp)
            elif method == "regex":
                chunks = extract_chunks_with_regex(text, pattern)
            else:
                chunks = text
            context.append((text, chunks))
        return context

    def get_vectordb_info(self):
        print('Getting vectorDB results...')
        return self.vectordb.get()
