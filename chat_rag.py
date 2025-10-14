### Fixed RAG Components

from typing_extensions import List, TypedDict, Annotated
from typing import Optional, Literal
from utils.rag_utils import analyze_query
from utils.ollama_utils import get_chat_model, get_vector_store, get_rerank_model
from datetime import datetime
import ast
import json
import os
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from typing_extensions import List,TypedDict
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langchain.globals import set_verbose
from langchain.tools import tool
from typing import Optional
from langgraph.graph import StateGraph


class RAGState(MessagesState):
    source: str = "both"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class ChatRAG():

    def __init__(self, chat_model_args=None, embedding_model_args=None, use_prefiltering=False,prompting_args=None, reranker_args = None):
        # Fix: Handle default arguments properly

        if reranker_args is None:
            reranker_args = {}
            reranker_args['use_reranker'] = False

        if reranker_args['use_reranker']:
            self.rerank_model = get_rerank_model()
        else:
            self.rerank_model = None

        if embedding_model_args is None:
            self.vector_store = get_vector_store()
        else:
            self.vector_store = get_vector_store(**embedding_model_args)
            
        if chat_model_args is None:
            self.chat_model = get_chat_model()
        else:
            self.chat_model = get_chat_model(**chat_model_args)
        
        if prompting_args:
            self.system_message_content = prompting_args['rag_content_instructions']

        self.use_prefiltering = use_prefiltering
        self.current_state = None

    def _process_date_query(self, search_params):
        """Process date query parameters using separate fields."""
        try:
            time_period = search_params.get('time_period')
            time_query = search_params.get('time_query')
            day = search_params.get('day')
            month = search_params.get('month') 
            year = search_params.get('year')
            
            if not all([time_period, time_query, year]):
                return None
            
            if time_period == 'day' and day and month:
                target_date = datetime(year, month, day).date()
                return {
                    'type': 'day',
                    'date': target_date,
                    'time_query': time_query
                }
            elif time_period == 'month' and month:
                return {
                    'type': 'month',
                    'month': month,
                    'year': year,
                    'time_query': time_query
                }
            elif time_period == 'year':
                return {
                    'type': 'year',
                    'year': year,
                    'time_query': time_query
                }
        except ValueError as e:
            print(f'Could not load date: {e}')
            return None

    def _date_filter_logic(self, date_criteria):
        if not date_criteria:
            return {}
            
        filter_type = date_criteria['type']
        time_query = date_criteria['time_query']
        
        if filter_type == 'day':
            target_date = int(date_criteria['date'].strftime("%Y%m%d"))
            if time_query == 'before':
                return {'end_date': {'$lt': target_date}}  
            elif time_query == 'after':
                return {'start_date': {'$gt': target_date}}  
            elif time_query == 'within':
                return {"$and": [
                    {"start_date": {"$lte": target_date}},  
                    {"end_date": {"$gte": target_date}},    
                ]}
        
        elif filter_type == 'month':
            month = date_criteria['month']
            year = date_criteria['year']
            start_int = int(f"{year}{month:02d}01")
            # Get last day of month
            if month == 12:
                end_int = int(f"{year}{month:02d}31")
            else:
                next_month = datetime(year, month + 1, 1) 
                end_int = int(next_month.strftime("%Y%m%d"))
            
            if time_query == 'before':
                return {'end_date': {'$lt': start_int}}
            elif time_query == 'after':
                return {'start_date': {'$gte': end_int}}
            elif time_query == 'within':
                return {"$and": [
                    {"start_date": {"$lte": end_int}},
                    {"end_date": {"$gte": start_int}},
                ]}
        
        elif filter_type == 'year':
            year = date_criteria['year']
            start_int = int(f"{year}0101")
            end_int = int(f"{year}1231")
            
            if time_query == 'before':
                return {'end_date': {'$lt': start_int}}
            elif time_query == 'after':
                return {'start_date': {'$gt': end_int}}
            elif time_query == 'within':
                return {"$and": [
                    {"start_date": {"$lte": end_int}},
                    {"end_date": {"$gte": start_int}},
                ]}
        
        return {}

    def _participant_filter_logic(self, search_params):
        """Generate participant filter for ChromaDB."""
        if not search_params.get('participants'):
            return {}
        
        # For multiple participants, use $or
        if len(search_params['participants']) == 1:
            return {"other_person": search_params['participants'][0]}
        else:
            return {"$or": [{"other_person": p} for p in search_params['participants']]}

    def _retrieve_with_prefiltering(self, date_criteria, search_params):
        """Use pre-filtering instead of post-filtering."""
        
        # Build ChromaDB filter
        filters = []
        
        # Add date filter
        date_filter = self._date_filter_logic(date_criteria)
        print(f'Created Date Filter : {date_filter}!')
        if date_filter:
            filters.append(date_filter)
        
        # Add participant filter  
        participant_filter = self._participant_filter_logic(search_params)
        if participant_filter:
            filters.append(participant_filter)
        
        # Combine filters
        if len(filters) == 0:
            chroma_filter = None
        elif len(filters) == 1:
            chroma_filter = filters[0]
        else:
            chroma_filter = {"$and": filters}
        
        # Use pre-filtering with ChromaDB
        retrieved_docs = self.vector_store.similarity_search(
            search_params['query'],
            k=3,  # Can use smaller k since pre-filtered
            filter=chroma_filter
        )
        return retrieved_docs

    def _retrieve_with_postfiltering(self, date_criteria, search_params):
        """Post-filtering approach - fetch docs then filter with exact same logic as pre-filtering."""
        # Get more documents to filter from
        all_docs = self.vector_store.similarity_search(
            search_params['query'],
            k=25,  # Get more docs to filter from
        )
        
        filtered_docs = []
        for doc in all_docs:
            metadata = doc.metadata
            include_doc = True
            
            # Apply participant filtering - exact same logic as pre-filtering
            if search_params.get('participants'):
                other_person = metadata.get('other_person', '')
                if len(search_params['participants']) == 1:
                    # Single participant: exact match
                    include_doc = other_person == search_params['participants'][0]
                else:
                    # Multiple participants: any match
                    include_doc = other_person in search_params['participants']
            
            # Apply date filtering - exact same logic as pre-filtering
            if include_doc and date_criteria:
                start_date_int = metadata.get('start_date')
                end_date_int = metadata.get('end_date')
                
                if start_date_int is None or end_date_int is None:
                    include_doc = False
                else:
                    filter_type = date_criteria['type']
                    time_query = date_criteria['time_query']
                    
                    if filter_type == 'day':
                        target_date = int(date_criteria['date'].strftime("%Y%m%d"))
                        if time_query == 'before':
                            include_doc = end_date_int < target_date
                        elif time_query == 'after':
                            include_doc = start_date_int > target_date
                        elif time_query == 'within':
                            include_doc = start_date_int <= target_date and end_date_int >= target_date
                    
                    elif filter_type == 'month':
                        month = date_criteria['month']
                        year = date_criteria['year']
                        start_int = int(f"{year}{month:02d}01")
                        # Use exact same month end calculation as pre-filtering
                        if month == 12:
                            end_int = int(f"{year}{month:02d}31")
                        else:
                            next_month = datetime(year, month + 1, 1)
                            end_int = int(next_month.strftime("%Y%m%d"))
                        
                        if time_query == 'before':
                            include_doc = end_date_int < start_int
                        elif time_query == 'after':
                            include_doc = start_date_int >= end_int
                        elif time_query == 'within':
                            include_doc = start_date_int <= end_int and end_date_int >= start_int
                    
                    elif filter_type == 'year':
                        year = date_criteria['year']
                        start_int = int(f"{year}0101")
                        end_int = int(f"{year}1231")
                        
                        if time_query == 'before':
                            include_doc = end_date_int < start_int
                        elif time_query == 'after':
                            include_doc = start_date_int > end_int
                        elif time_query == 'within':
                            include_doc = start_date_int <= end_int and end_date_int >= start_int
            
            if include_doc:
                filtered_docs.append(doc)
                if len(filtered_docs) >= 5:
                    break
        
        return filtered_docs

    def create_retrieve_tool(self):
        """Create a retrieve tool that wraps the instance method."""
        
        def retrieve_wrapper(query: str) :
            serialized, retrieved_docs= self.retrieve(query)  # Uses captured 'self'
            return  serialized, retrieved_docs

        # Apply @tool decorator to the wrapper (no 'self' parameter!)
        @tool(response_format="content_and_artifact")  
        def retrieve_tool(query: str): 
            """Retrieve information related to a query with intelligent filtering.
              Use the retrieve tool for:
            - Questions about past conversations, messages, or interactions
            - Personal topics, preferences, activities, or experiences mentioned in chats
            - Requests about discussions with specific people or groups
            - Any question about 'my', 'our', or personal history/behavior.


            Args:
                query : Query to use tool for.
            """
            return retrieve_wrapper(query)
        
        self.retrieve_tool = retrieve_tool
        return retrieve_tool

    def process_manual_filtering(self):
        state = self.current_state
        start_date = state.get("start_date", None)
        end_date = state.get("end_date", None)
        source = state.get("source", None)
        print(f'SOURCE {source}')

        filters= []
        if start_date and end_date:
            filters.append(
                 {"$and" : [
                    {"start_date": {"$gte": int(start_date.strftime("%Y%m%d"))}},  
                    {"end_date": {"$lte": int(end_date.strftime("%Y%m%d"))}},]}
            )
        if source is not None and source!="both":
           filters.append({"source": source})

        if len(filters) == 0:
            chroma_filter = None
        elif len(filters) == 1:
            chroma_filter = filters[0]
        else:
            chroma_filter = {"$and": filters}

        return chroma_filter


    def retrieve(self, query: str):
        """Retrieve information related to a query with intelligent filtering."""
        print(f"\n🔍 RETRIEVE DEBUG - Query: '{query}'")
        
        # Fix: Pass chat_model to analyze_query
        # search_params = analyze_query(self.chat_model, query)
        # date_criteria = self._process_date_query(search_params)
        # # Check if filtering is needed
        # needs_filtering = date_criteria or search_params.get('participants')
        # print(f'HERE SET NEEDS FILTERING TO FALSE!!')
        

        date_criteria = None
        search_params = None
        needs_filtering = None
        chroma_filter = self.process_manual_filtering()

        
        if needs_filtering:
            if self.use_prefiltering:
                print("Using PRE-FILTERING approach")
                retrieved_docs = self._retrieve_with_prefiltering(date_criteria, search_params)
            else:
                print("Using POST-FILTERING approach")
                # Fix: Add missing method name
                retrieved_docs = self._retrieve_with_postfiltering(date_criteria, search_params)
        elif self.rerank_model:
            # No filtering needed
            print(f'Chroma Filter : {chroma_filter}')
            retrieved_docs = self.vector_store.similarity_search(
                query,
                k=16,
                filter = chroma_filter
            )

            retrieved_docs = self.rerank_model.batch_rank_chunks(query=query,chunks=retrieved_docs,top_k=3,batch_size=16)['chunks']
        else:
            retrieved_docs = self.vector_store.similarity_search(
                query,
                k=3,
                filter = chroma_filter
            )

        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('chat_name', 'Unknown')}\n"
            f"Date Range: {doc.metadata.get('date_range', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        
        return serialized, retrieved_docs

    ### Setup LangGraph components

    def _invoke_llm_with_retrieve_tool(self, messages, system_message: str, tool_choice : str):
        """Common method to invoke LLM with retrieve tool and system message."""
        llm_with_tools = self.chat_model.bind_tools([self.retrieve_tool], tool_choice = tool_choice)
        system_msg = SystemMessage(system_message)
        messages_with_system = [system_msg] + messages
        response = llm_with_tools.invoke(messages_with_system)

        return {"messages": [response]}

    def manual_decider(self, state: RAGState):
        
        # keep current state in memory
        self.current_state = state


        user_message = state["messages"][-1].content.lower()
        #trigger_words = ['chat', 'message', 'messages', 'my', 'mine', 'our', 'conversation','retrieve','conversations','friends','friend']
        trigger_words = ['rag', 'retrieve', 'tool-call']
        if any(word in user_message for word in trigger_words):
            system_message = """
            You are a RAG chat model with access to WhatsApp and Instagram chat history.
            ALWAYS use the retrieve tool for every user query - no exceptions.
            Never respond directly without first calling the retrieve tool.
            """

            clean_messages = [msg for msg in state["messages"] if msg.type!="tool"]

            return self._invoke_llm_with_retrieve_tool(clean_messages, system_message, tool_choice = 'retrieve_tool')
        else:
            return {"messages": state["messages"]}
    

    def query_or_respond(self, state: RAGState):
        """Generate tool call for retrieval or direct respond."""
        system_message = """
        You are a RAG chat model with access to WhatsApp and Instagram chat history. Use the retrieve tool for:
        - Questions about past conversations, messages, or interactions
        - Personal topics, preferences, activities, or experiences mentioned in chats
        - If context if necessary context is already present in chat history, you must not call the tool
        - Any question about 'my', 'our', or personal history/behavior, if not already in chat history


        You do not know anything about the user, only what is in the context, do no hallucinate things not in the context, but call the retrieve tool.
        Only respond directly for pure general knowledge unrelated to personal chat content.
        """
        
        return self._invoke_llm_with_retrieve_tool(state["messages"], system_message, tool_choice = 'auto')


    def generate(self, state: RAGState):
        recent_tool_messages = []

        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        system_message_content = self.system_message_content.replace('<docs_content>',docs_content)
        conversation = [
            message
            for message in state["messages"]
            if message.type in ("human","system")
            or (message.type =="ai" and not message.tool_calls)
        ]

        prompt = [SystemMessage(system_message_content)] + conversation   

        print(f'Prompt to model : {prompt}\n ******************************************************')
        response = self.chat_model.invoke(prompt) 
        return {"messages": [response]}
    


    def store_messages(self, state: RAGState):
        """Store messages, overwriting any existing data."""
        
        # Create new data structure (overwrites any exi
        # 
        # sting data)
        file_path = "data/evaluation/messages_log.json"
        folder = os.path.dirname(file_path)
        os.makedirs(folder,exist_ok=True)


        data = [{"type": msg.type, "content": msg.content} for msg in state["messages"] if msg.type != "tool"]
        
        # Save to file (completely overwrites existing file)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {"messages": state["messages"]}
    
    
            
    def initialize_graph_builder(self):
        set_verbose(True)

        self.create_retrieve_tool()  
        tools = ToolNode([self.retrieve_tool])

        graph_builder = StateGraph(RAGState)
        # manual pre-emptive
        graph_builder.add_node("manual_decider",self.manual_decider)
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_node("store_messages", self.store_messages)

        #graph_builder.set_entry_point("query_or_respond")
        graph_builder.set_entry_point("manual_decider")
        graph_builder.add_conditional_edges(
            "manual_decider",
            tools_condition,
            {END: "query_or_respond", "tools": "tools"}

        )
        graph_builder.add_conditional_edges( 
            "query_or_respond",
            tools_condition,
            {END: "store_messages", "tools": "tools"}
        )

        
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", "store_messages")
        graph_builder.add_edge("store_messages",END)

        return graph_builder

    def initialize_graph(self):
        graph_builder = self.initialize_graph_builder()

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory,debug=False)
        # specify thread_id
        config = {"configurable": {"thread_id": "abc123"}}
        
        # Fix: Return both graph and config
        return graph, config