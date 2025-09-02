from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from langchain_core.messages import HumanMessage
from utils.ollama_utils import format_response_metadata
import time
import uuid
from chat_rag import ChatRAG
from datetime import datetime
import atexit

import os
import requests
import subprocess


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to store the RAG instance and graph
rag_system = None
rag_graph = None
rag_config = None
verbose_response = True

ORIGINAL_PROMPT = """
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
        """

CONTEXT_AWARENESS_PROMPT = """
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
        "- Even if exceprts are in another langauge, answer in english.\n"
        "- If information is incomplete, acknowledge what's missing\n\n"
        "Your chat history excerpts:\n"
        "<docs_content>\n\n"
        "Based on this context, what can you tell the user?"
"""
LLAMACPP_URL = os.getenv("LLAMACPP_BASE_URL","http://localhost:10000")

llamacpp_process = None
def launch_llama_server():
    global llamacpp_process
    try:
        requests.get(LLAMACPP_URL+"/health", timeout=1)
        return True
    except:
        llamacpp_process = subprocess.Popen(["/opt/homebrew/bin/llama-server", "--model", 
                         "/Users/pariidan/Documents/python_projects/RAG_Langchain/models/Qwen3-Reranker-0.6B-q4_k_m.gguf",
                         "--port", "10000", "--ctx-size", "4096", "--cont-batching"], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(15):
            time.sleep(1)
            try:
                requests.get(LLAMACPP_URL+"/health", timeout=1)
                return True
            except:
                continue
        return False
    
def cleanup():
    if llamacpp_process:
        llamacpp_process.terminate()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_rag():
    global rag_system, rag_graph, rag_config
    
    try:
        data = request.json
        chat_model = data.get('chat_model', 'qwen3:14b')
        embedding_model = data.get('embedding_model', 'Qwen3-Embedding-4B-Q4KM:latest')

        use_prefiltering = data.get('use_prefiltering', False)
        use_reranker = data.get('use_reranker', True)
        if use_reranker and not launch_llama_server():
            return jsonify({'status': 'error', 'message': 'Failed to start reranker server'}), 500
        
        # Initialize RAG system
        rag_system = ChatRAG(
            use_prefiltering=use_prefiltering,
            chat_model_args={'model_name': chat_model,'context_len': 4096},
            embedding_model_args={'model_name': embedding_model,'context_len': 2048},
            reranker_args={"use_reranker": use_reranker},
            prompting_args={"rag_content_instructions": CONTEXT_AWARENESS_PROMPT}
        )
        
        # Store graph and config globally instead of in session
        rag_graph, rag_config = rag_system.initialize_graph()
        
        return jsonify({
            'status': 'success',
            'message': f'RAG system initialized with {chat_model} and {embedding_model}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to initialize: {str(e)}'
        }), 500

@socketio.on('send_message')
def handle_message(data):
    global rag_graph, rag_config
    
    if not rag_system or not rag_graph:
        emit('response', {
            'type': 'error',
            'content': 'RAG system not initialized. Please configure first.'
        })
        return
    
    try:
        question = data['message']
        session_id = data.get('session_id', str(uuid.uuid4()))
        source = data.get('source', 'both')
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)
        
        # Parse dates
        def parse_date(date_str, ):
            if not date_str: return None
            try:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    print(f"Warning: Could not parse {date_str}")
                    return None
            return dt
        
        start_datetime = parse_date(start_date)
        end_datetime = parse_date(end_date)
        
        print(f'Source: {source} | Start: {start_datetime} | End: {end_datetime}')
        messages = [HumanMessage(content=question)]

           # Create message
        graph_input = {"start_date" : start_datetime,
                       "end_date" : end_datetime,
                       "source" : source,
                       "messages" : messages
                       }
        
        # Use the global graph and config
        config = rag_config.copy()
        config['configurable']['thread_id'] = session_id
        
        emit('response', {'type': 'start', 'content': ''})
        
        # Stream response from graph
        full_response = ""
        context_docs = []
        
        for step in rag_graph.stream(
            graph_input,
            stream_mode="values",
            config=config,
        ):
            if step["messages"]:
                last_message = step["messages"][-1]

                if verbose_response and last_message.type=='ai':
                    print(f'Message: {last_message.content[:30]}..')
                    print(format_response_metadata(last_message))
                    print('-'*50)
                
                # Handle tool messages (context)
                if last_message.__class__.__name__ == 'ToolMessage':
                    context_docs.append(last_message.content)
                    emit('response', {
                        'type': 'context',
                        'content': last_message.content
                    })
                
                elif (hasattr(last_message, 'content') and 
                    last_message.__class__.__name__ == 'AIMessage' and 
                    last_message.content and
                    not getattr(last_message, 'tool_calls', None)):  # Exclude AI messages with tool calls
                    
                    # Send incremental content
                    new_content = last_message.content
                    if new_content != full_response:
                        chunk = new_content[len(full_response):]
                        full_response = new_content
                        
                        emit('response', {
                            'type': 'chunk',
                            'content': chunk
                        })
        
        # Send completion signal
        emit('response', {
            'type': 'complete',
            'content': '',
            'context': context_docs
        })
        
    except Exception as e:
        emit('response', {
            'type': 'error',
            'content': f'Error: {str(e)}'
        })

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

atexit.register(cleanup)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000,debug=True,allow_unsafe_werkzeug=True)