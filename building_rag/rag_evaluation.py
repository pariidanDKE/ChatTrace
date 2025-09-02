import pandas as pd
import json
import os
import time
from datetime import datetime
from langchain_core.messages import HumanMessage
from chat_rag import ChatRAG

# Updated strategy dictionary with your format
rag_strategies = {
    "original_prompt": (
        "You are an assistant for answering questions about the user's WhatsApp or Instagram chat history.\n\n"
        "Instructions:\n"
        "- Answer using the provided chat excerpts only\n"
        "- Be specific: mention who said what and when (names, dates)\n"
        "- If no relevant information exists, say 'I don't have information about that'\n"
        "- If no excerpts show 'Me' participating, assume the user wasn't involved\n"
        "- Keep responses conversational, add specific details and quotes from the message if appropriate.\n"
        "- Do not exceed 5 sentences.\n"
        "- Never make up facts.\n\n"
        "Here is the some chats that are potentially relevant to the query:\n"
        "<docs_content>"
    ),
    
    "condensed_query_context": (
        "You are an assistant for answering questions about WhatsApp or Instagram chat history.\n\n"
        "Step 1: First, identify the core meaning of the user's question.\n"
        "Step 2: Use only the chat excerpts below to form your answer.\n\n"
        "Instructions:\n"
        "- Be specific: mention who said what and when (names, dates)\n"
        "- Quote directly from messages when relevant\n"
        "- Keep responses conversational and under 5 sentences\n"
        "- If no excerpts show 'Me' participating, assume the user wasn't involved\n\n"
        "Chat excerpts from your history:\n"
        "<docs_content>\n\n"
        "Now provide the best answer you can based on these excerpts."
    ),
    
    "chain_of_thought": (
        "You are an assistant for answering questions about WhatsApp or Instagram chat history.\n\n"
        "Follow these steps:\n"
        "Step 1: Summarize the user's question in simple terms\n"
        "Step 2: Identify which chat excerpts directly relate to their question\n"
        "Step 3: Extract key details (who, what, when) from relevant excerpts\n"
        "Step 4: Provide a clear, conversational answer\n\n"
        "Guidelines:\n"
        "- Quote specific messages when helpful\n"
        "- Mention names and dates from the chats\n"
        "- Keep final answer under 5 sentences\n"
        "- Never fabricate information\n\n"
        "Available chat excerpts:\n"
        "<docs_content>\n\n"
        "Show your reasoning steps, then give your final answer."
    ),
    
    "dont_know_scenarios": (
        "You are an assistant for answering questions about WhatsApp or Instagram chat history.\n\n"
        "Below are potentially relevant chat excerpts. If these excerpts contain enough "
        "information to answer the user's question completely, provide a detailed response. "
        "If not, respond: 'I don't have enough information about that in your chat history.'\n\n"
        "When answering:\n"
        "- Be specific about who said what and when\n"
        "- Include direct quotes from messages when relevant\n"
        "- Keep responses conversational and under 5 sentences\n"
        "- If 'Me' doesn't appear in excerpts, assume user wasn't involved\n\n"
        "Chat excerpts:\n"
        "<docs_content>\n\n"
        "What is your final answer based on this evidence?"
    ),
    
    "critique_and_revision": (
        "You are an assistant for answering questions about WhatsApp or Instagram chat history.\n\n"
        "Task: Produce a thorough answer using the chat excerpts below, then revise it.\n\n"
        "Step 1: Create an initial answer incorporating all relevant chat details\n"
        "Step 2: Review your answer - did you miss any important context from the excerpts?\n"
        "Step 3: Provide a final, improved version\n"
        "Step 4: List which specific chat excerpts you used\n\n"
        "Guidelines:\n"
        "- Include names, dates, and direct quotes\n"
        "- Be conversational but precise\n"
        "- Keep final answer under 5 sentences\n"
        "- If 'Me' isn't in excerpts, note user wasn't involved\n\n"
        "Available chat excerpts:\n"
        "<docs_content>\n\n"
        "Show your initial answer, revision, and final response with sources used."
    ),
    
    "context_awareness": (
        "You are an assistant for answering questions about WhatsApp or Instagram chat history.\n\n"
        "Before answering, analyze the chat excerpts below to understand:\n"
        "- Who are the main participants in relevant conversations?\n"
        "- What timeframes are covered?\n"
        "- What topics or events are being discussed?\n\n"
        "Then provide your answer following these rules:\n"
        "- Use only information explicitly present in the excerpts\n"
        "- Quote specific messages when they directly address the question\n"
        "- Mention participant names and dates/times\n"
        "- Be conversational but accurate (max 5 sentences)\n"
        "- If excerpts don't contain 'Me', assume user wasn't part of those conversations\n"
        "- If information is incomplete, acknowledge what's missing\n\n"
        "Your chat history excerpts:\n"
        "<docs_content>\n\n"
        "Based on this context, what can you tell the user?"
    ),
    
    "context_awareness_enhanced": {
        "You are an assistant for answering questions about WhatsApp/Instagram chat history.\n\n"
        
        "Before answering, analyze the chat excerpts to identify:\n"
        "- Main participants in relevant conversations\n"
        "- Covered timeframes\n"
        "- Topics/events discussed\n"
        "- Any ambiguous terms that could be misinterpreted\n\n"
        
        "Answer following these rules:\n"
        "- Use ONLY explicitly stated information from excerpts (no inference)\n"
        "- Always include participant names and exact dates/times\n"
        "- State the type/category of any entity mentioned (stock, game, event, etc.)\n"
        "- Quote messages that directly address the question\n"
        "- If excerpts lack 'Me', state the user was not part of those conversations\n"
        "- If information is incomplete/ambiguous, clearly say so\n"
        "- Keep responses conversational but accurate, max 4 sentences\n\n"
        
        "Quality checks before responding:\n"
        "- Only verifiable facts from excerpts\n"
        "- Correct identification of entity types\n"
        "- Explicitly state WHO said WHAT and WHEN\n\n"
        
        "Your chat history excerpts:\n"
        "<docs_content>\n\n"
        
        "Based on this context, provide a precise and informative answer."
    },

    
    "verification_approach": (
        "You are an assistant for WhatsApp or Instagram chat analysis.\n\n"
        "Your job: Answer questions using ONLY the chat excerpts provided below.\n\n"
        "Response format:\n"
        "1. If excerpts fully answer the question → Provide specific details with quotes\n"
        "2. If excerpts partially answer → Share what you found + note what's missing\n"
        "3. If excerpts don't answer → Say 'No relevant information in your chat history'\n\n"
        "Always include:\n"
        "- Names of people involved\n"
        "- Dates/times when available\n"
        "- Direct quotes when helpful\n"
        "- Note if 'Me' appears (otherwise assume user wasn't involved)\n\n"
        "Keep responses conversational, accurate, and under 5 sentences.\n\n"
        "Chat excerpts to analyze:\n"
        "<docs_content>"
    )
}

def test_rag_strategies(use_prefiltering=False,use_reranker=False):
    """Test all RAG strategies with predefined queries and stream results to JSON file."""
    
    # # Test queries
    # test_queries = [
    # "What work activities did Cristina and I discuss in our WhatsApp messages?",
    # "Which musicians did I mention in my chats and what did I say about why I like them?", 
    # "Show me discussions from my messages about gym and workout planning with friends",
    # "What did Teodor Lunugu and I talk about regarding religion and praying in our conversations?",
    # "What daily activities did I tell Cristina about in our recent messages?"
    # ]

    test_queries = [
    "Which stocks did I talk to investing in with others",
    "What events did I plan to watch together with my friends", 
    "Did I play any games with other people recently? What games and which people?",
    "What homework projects am I doing at my Data Science University, who do I talk to about it?",
    "Discussing grades on recent exams and projects",
    "Talking about university assignments and issues with them"
    ]
    
    json_path = '/Users/pariidan/Documents/python_projects/RAG_Langchain/data/evaluation/rag_strategy_responses.json'
    
    print("🧪 Testing RAG strategies...")
    print(f"📊 {len(rag_strategies)} strategies × {len(test_queries)} queries = {len(rag_strategies) * len(test_queries)} total tests")
    print(f"📁 Results will be saved to: {json_path}")
    print("-" * 60)
    
    # Load existing results if file exists
    all_results = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
                print(f"📋 Loaded {len(all_results)} existing results")
            except json.JSONDecodeError:
                print("📋 Starting with empty results (invalid JSON file)")
                all_results = []
    
    for strategy_name, strategy_prompt in rag_strategies.items():
        print(f"\n🔬 Testing strategy: {strategy_name}")
        
        try:
            # Initialize new RAG system with this strategy
            rag = ChatRAG(
                use_prefiltering=use_prefiltering,
                chat_model_args={'model_name': 'qwen3:8b'},
                embedding_model_args={'model_name': 'Qwen3-Embedding-4B-Q4KM:latest'},
                prompting_args={"rag_content_instructions":strategy_prompt},
                reranker_args={'use_reranker' : use_reranker}
            )
            graph, config = rag.initialize_graph()
            
            for i, query in enumerate(test_queries, 1):
                print(f"  📝 Query {i}/{len(test_queries)}: {query[:50]}...")
                
                start_time = time.time()
                timestamp = datetime.now().isoformat()
                
                try:
                    # Create message and get response
                    messages = [HumanMessage(content=query)]
                    
                    # Collect response from stream and capture tool messages
                    response_parts = []
                    tool_contexts = []
                    for step in graph.stream(
                        {"messages": messages},
                        stream_mode="values",
                        config=config,
                    ):
                        if step["messages"]:
                            last_message = step["messages"][-1]
                            if hasattr(last_message, 'content'):
                                if last_message.__class__.__name__ == 'ToolMessage':
                                    # Capture tool message content (retrieved context)
                                    tool_contexts.append(last_message.content)
                                else:
                                    # Capture regular response content
                                    response_parts.append(last_message.content)
                    
                    response = ''.join(response_parts)
                    retrieved_context = '\n---\n'.join(tool_contexts) if tool_contexts else None
                    end_time = time.time()
                    response_time = end_time - start_time
                    error = None
                    
                    print(f"    ✅ Completed in {response_time:.2f}s")
                    
                except Exception as e:
                    response = f"ERROR: {str(e)}"
                    end_time = time.time()
                    response_time = end_time - start_time
                    error = str(e)
                    retrieved_context = None
                    print(f"    ❌ Error: {str(e)}")
                
                # Create result dictionary
                result = {
                    'strategy': strategy_name,
                    'query': query,
                    'response': response,
                    'retrieved_context': retrieved_context,
                    'timestamp': timestamp,
                    'response_time': response_time,
                    'use_prefiltering': use_prefiltering,
                    'error': error,
                    'use_reranker' : use_reranker
                }
                
                # Add to results list
                all_results.append(result)
                
                # Save results to JSON file after each query (incremental save)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            # Handle strategy initialization errors
            print(f"  ❌ Strategy initialization failed: {str(e)}")
            for query in test_queries:
                result = {
                    'strategy': strategy_name,
                    'query': query,
                    'response': f"STRATEGY_INIT_ERROR: {str(e)}",
                    'retrieved_context': None,
                    'timestamp': datetime.now().isoformat(),
                    'response_time': 0.0,
                    'use_prefiltering': use_prefiltering,
                    'error': f"Strategy init error: {str(e)}",
                    'use_reranker' : use_reranker
                }
                
                all_results.append(result)
                
                # Save after each error too
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Testing complete! Results saved to {json_path}")
    print(f"📊 Total results: {len(all_results)}")
    
    # Return results as DataFrame for compatibility if needed
    return pd.DataFrame(all_results)

def load_results_as_dataframe(json_path='/Users/pariidan/Documents/python_projects/RAG_Langchain/data/evaluation/rag_strategy_responses.json'):
    """Load JSON results and convert to DataFrame for analysis."""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return pd.DataFrame(results)
    else:
        print(f"No results file found at {json_path}")
        return pd.DataFrame()

# Run the test
if __name__ == "__main__":
    test_rag_strategies(use_reranker=True)