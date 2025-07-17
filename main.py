import pandas as pd
from tqdm import tqdm

from RAG import RAG

k = 3  # Number of results to retrieve

data_folder = '2WikiMultihopQA'  # '2WikiMultihopQA', 'HotpotQA', 'CloudComputing'
data_extenstion = '-full-list'  # '-full-list', '-dev-context'
llm_provider = 'gemini'  # 'openai', 'ollama', 'huggingface', 'gemini'
mode = 'rag+related_data'  # 'related_data', 'rag+related_data', 'rag&related_data', 'rag'

rag = RAG(data_folder=data_folder,
          embedding_model='all-MiniLM-L6-v2',
          chromadb_extension_name=data_extenstion,
          mode=mode,
          llm_provider=llm_provider,
          llm_model="default",
          additional_results_name='MiniLM-L6-150/')
rag.start(k=k)

results = rag.load_results()
questions = rag.load_questions()
total_question = len(questions)
print(questions)

for index, row in tqdm(questions.iterrows(),
                       desc='Getting results',
                       dynamic_ncols=True,
                       total=total_question):
    res = rag.run(row['question'])['result']
    retrieval_context = rag.get_context(row['question'], )

    if res == "Error: The model name or API request is invalid.":
        print(f"Error: {res}")
        continue

    new_row = pd.DataFrame({
        'id': [index],
        'type': [row['type']],
        'question': [row['question']],
        'answer': [row['answer']],
        'result': [res],
        'k': [k],
        'retrieval_context': [retrieval_context],
        'llm_model': [rag.llm_model],
    })
    results = pd.concat([results, new_row], ignore_index=True)

rag.save_results(results=results)
print('results saved')
