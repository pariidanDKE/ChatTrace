import requests
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class Reranker():
    def __init__(self, model_name='Qwen/Qwen3-Reranker-0.6B',model = None,llama_cpp = False):
        self._init_hyperparams()
        self.llama_cpp = llama_cpp
        self.LLAMACPP_URL = os.getenv("LLAMACPP_BASE_URL","http://localhost:10000")

    
    def _init_hyperparams(self):
        self.max_length = 2048
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    def _format_instructions(self, query, doc, instruction=None):
        if instruction is None:
            instruction = "Given a question about specific details from chats of users, retrieve relevant chat passages that answer the query."
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output
    
    
    def batch_rank_chunks(self, query, chunks, batch_size=16,top_k=3):
        instruction = "Given a question about specific details from chat excerpts, retrieve relevant passages that answer the query."
        pairs = [self._format_instructions(instruction=instruction, query=query, doc=doc.page_content) for doc in chunks]
        step = min(batch_size, len(pairs))
        
        if self.llama_cpp:
            return self.batch_rank_llama_cpp( query, chunks,instruction=instruction,batch_size=16,top_k=3,pairs=pairs,step=step)
        else:
            # completley remove the torch/transformers implementation for now ( clutters up image)
            return None
        
    def batch_rank_llama_cpp(self, query, chunks, instruction, batch_size=16, top_k=3,step=16,pairs=None):
        """Rerank chunks using llama cpp with continuous batching for faster inference."""
        def query_server(prompt, i):
            url = self.LLAMACPP_URL + "/completion"
            resp = requests.post(url, 
                                json={"prompt": prompt, "n_predict": 1, "n_probs": 3, "temperature": 0, "stream": False})
            if resp.ok:
                data = resp.json()
                return i, data['content'].strip().lower(), math.exp((data['completion_probabilities'][0]['logprob']))
            return None
        

        all_results = []
        for i in range(0, len(pairs), step):
            batch_pairs = pairs[i:i+batch_size]
            inputs = [self.prefix + pair + self.suffix for pair in batch_pairs]
            
            with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
                for f in as_completed([executor.submit(query_server, prompt, j) for j, prompt in enumerate(inputs)]):
                    if (r := f.result()) is not None:
                        all_results.append((i + r[0], r[1], r[2]))
        
        yes = [(idx, prob) for idx, content, prob in all_results if content == 'yes']
        if yes:
            sorted_yes = sorted(yes, key=lambda x: x[1], reverse=True)
            top_results = sorted_yes[:top_k]
        else:
            no_results = [(idx, prob) for idx, content, prob in all_results if content == 'no']
            sorted_no = sorted(no_results, key=lambda x: x[1])
            top_results = sorted_no[:top_k]

        final_indices = [idx for idx, _ in top_results]
        top_chunks = [chunks[i] for i in final_indices]

        final_scores = [score for _, score in top_results]
        final_ids = [doc.id for doc in top_chunks]
        
        return {
            'chunks' : top_chunks,
            'scores' : final_scores,
            'ids' : final_ids,
        }
                


