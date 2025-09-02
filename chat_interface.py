from chat_rag import ChatRAG
from langchain_core.messages import HumanMessage


ORIGINAL_PROPMT = """
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


class ChatInterface:
    """Simple interface for asking multiple questions to the RAG system."""
    
    def __init__(self, use_prefiltering=False,use_reranker=False):
        """Initialize the chat interface with RAG system."""
        self.rag = ChatRAG(use_prefiltering=use_prefiltering,chat_model_args={'model_name' : 'qwen3:14b'},embedding_model_args={'model_name' : 'Qwen3-Embedding-4B-Q4KM:latest'},reranker_args={"use_reranker" : True}
                           ,prompting_args={"rag_content_instructions":ORIGINAL_PROPMT})
        self.graph, self.config = self.rag.initialize_graph()
        print("🤖 RAG Chat Interface initialized!")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print("-" * 50)
    
    def ask(self, question: str):
        """Ask a single question and stream response."""
        try:
            # Create message
            messages = [HumanMessage(content=question)]

            # Stream response from graph
            for step in self.graph.stream(
                {"messages": messages},
                stream_mode="values",
                config=self.config,
            ):
                if step["messages"]:
                    # Get the last message and print it
                    last_message = step["messages"][-1]
                          
                    if hasattr(last_message, 'content') and last_message.__class__.__name__ != 'ToolMessage':
                        print(last_message.content, end='', flush=True)
                        
        except Exception as e:
            print(f"Error: {str(e)}")

    def chat_loop(self):
        """Interactive chat loop with streaming responses."""
        while True:
            try:
                # Get user input
                question = input("\n❓ Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                
                # Stream response
                print("\n🤖 Response:")
                self.ask(question)
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
        
    def ask_batch(self, questions: list) -> list:
        """Ask multiple questions in batch and return all responses."""
        responses = []
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}: {question}")
            response = self.ask(question)
            print(f"🤖 Response {i}: {response}")
            responses.append({"question": question, "response": response})
            print("-" * 30)
        return responses


# Example usage functions
def run_interactive_chat(use_prefiltering=False,use_reranker=False):
    """Start an interactive chat session."""
    chat = ChatInterface(use_prefiltering=use_prefiltering,use_reranker=use_reranker)
    chat.chat_loop()

def ask_single_question(question: str, use_prefiltering=False):
    """Ask a single question and return response."""
    chat = ChatInterface(use_prefiltering=use_prefiltering)
    return chat.ask(question)

def ask_multiple_questions(questions: list, use_prefiltering=False):
    """Ask multiple questions and return all responses."""
    chat = ChatInterface(use_prefiltering=use_prefiltering)
    return chat.ask_batch(questions)


# Main execution - directly starts interactive chat when file is run
if __name__ == "__main__":
    print("🚀 WhatsApp RAG Chat Interface")
    print("Ask questions aboutf your WhatsApp conversations!")
    print("=" * 50)
    
    # Start interactive chat immediately
    run_interactive_chat(use_prefiltering=False,use_reranker=True)