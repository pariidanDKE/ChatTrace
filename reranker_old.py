from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


"""
This Reranker implementation uses trasnformers through hf models, it also needs torch then.
To save space in the docker image this version has been deprecated and only the separate llama-cpp one remains.
"""
class Reranker():
    def __init__(self, model_name='Qwen/Qwen3-Reranker-0.6B',model = None,llama_cpp = False):
        self._init_tokenizer()
        self._init_hyperparams()
        self.llama_cpp = llama_cpp
        if not llama_cpp:
            self.model_name = model_name
            self.device = 'mps'
            self._init_model(model)
      
    
    def _init_model(self,model):
        print(f'Model Name : {self.model_name}')
        if model:
            self.model = model.to(self.device)
        else:

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            ).to(self.device)
    
    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
    
    def _init_hyperparams(self):
        self.max_length = 2048
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
    
    def _format_instructions(self, query, doc, instruction=None):
        if instruction is None:
            instruction = "Given a question about specific details from chats of users, retrieve relevant chat passages that answer the query."
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output
    
    def _process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',  # longest first means it will shorten from the context chunk, not the query
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        print(f'Input IDs Shape: {inputs["input_ids"].shape}')

        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors='pt')
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rank_chunks(self, query, chunks, top_k=3):
        """  Example implementation, rank batch of size 1 with transformers """
        instruction = "Given a question about specific details from chat excerpts, retrieve relevant passages that answer the query."
        pairs = [self._format_instructions(instruction=instruction, query=query, doc=doc.page_content) for doc in chunks]
        ids = [doc.id for doc in chunks]
        inputs = self._process_inputs(pairs)
        scores = self.compute_logits(inputs)
        
        # Create list of (chunk, score, id) tuples and sort by score descending
        chunk_scores = list(zip(chunks, scores, ids))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k chunks with highest scores
        top_chunks = [item[0] for item in chunk_scores[:top_k]]
        top_scores = [item[1] for item in chunk_scores[:top_k]]
        top_ids = [item[2] for item in chunk_scores[:top_k]]
        
        return {
            'chunks': top_chunks,
            'scores': top_scores,
            'ids': top_ids
        }
    
    #### 
    def batch_rank_chunks(self, query, chunks, batch_size=16,top_k=3):
        instruction = "Given a question about specific details from chat excerpts, retrieve relevant passages that answer the query."
        pairs = [self._format_instructions(instruction=instruction, query=query, doc=doc.page_content) for doc in chunks]
        step = min(batch_size, len(pairs))
        
        if self.llama_cpp:
            return self.batch_rank_llama_cpp( query, chunks,instruction=instruction,batch_size=16,top_k=3,pairs=pairs,step=step)
        else:
            ### CHANGE TO BATCH RANK
            for i in range(0, len(pairs), step):
                batch_pairs = pairs[i : i + batch_size]
                batch_chunks = chunks[i : i + batch_size]
                ids = [doc.id for doc in batch_chunks]


                inputs = self._process_inputs(batch_pairs)
                scores = self.compute_logits(inputs)
            
                # Create list of (chunk, score, id) tuples and sort by score descending
                chunk_scores = list(zip(batch_chunks, scores, ids))
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Return top_k chunks with highest scores
                top_chunks = [item[0] for item in chunk_scores[:top_k]]
                top_scores = [item[1] for item in chunk_scores[:top_k]]
                top_ids = [item[2] for item in chunk_scores[:top_k]]
            
            return {
                'chunks': top_chunks,
                'scores': top_scores,
                'ids': top_ids
            }
        
        
    def batch_rank_llama_cpp(self, query, chunks, instruction, batch_size=16, top_k=3,step=16,pairs=None):
        """Rerank chunks using llama cpp with continuous batching for faster inference."""
        def query_server(prompt, i):
            resp = requests.post("http://localhost:10000/completion", 
                                json={"prompt": prompt, "n_predict": 1, "n_probs": 3, "temperature": 0, "stream": False})
            if resp.ok:
                data = resp.json()
                return i, data['content'].strip().lower(), torch.exp(torch.tensor(data['completion_probabilities'][0]['logprob'])).item()
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
                


