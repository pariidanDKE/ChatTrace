### Fixed RAG Components

from typing_extensions import List, TypedDict, Annotated
from typing import Optional, Literal
from rag_utils import analyze_query
from ollama_utils import get_chat_model, get_vector_store # singletons
from datetime import datetime
import ast
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from typing_extensions import List,TypedDict
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph


class ChatRAG():

    def __init__(self, chat_model_args=None, embedding_model_args=None, use_prefiltering=False,prompting_args=None):
        # Fix: Handle default arguments properly
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
        

        print(f'Chroma Filter : {chroma_filter}')
        # Use pre-filtering with ChromaDB
        retrieved_docs = self.vector_store.similarity_search(
            search_params['query'],
            k=4,  # Can use smaller k since pre-filtered
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
        
        def retrieve_wrapper(query: str) -> str:
            """Retrieve information related to a query with intelligent filtering."""
            serialized, retrieved_docs= self.retrieve(query)  # Uses captured 'self'
            return  serialized, retrieved_docs

        # Apply @tool decorator to the wrapper (no 'self' parameter!)
        from langchain.tools import tool
        
        @tool(response_format="content_and_artifact")  
        def retrieve_tool(query: str) -> str: 
            """Retrieve information related to a query with intelligent filtering."""
            return retrieve_wrapper(query)
        
        return retrieve_tool  # Returns the decorated function


    #@tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retrieve information related to a query with intelligent filtering."""
        print(f"\n🔍 RETRIEVE DEBUG - Query: '{query}'")
        
        # Fix: Pass chat_model to analyze_query
        search_params = analyze_query(self.chat_model, query)
        print(f"📊 Search params: {search_params}")
        
        date_criteria = self._process_date_query(search_params)
        print(f'Date Criteria {date_criteria}')
        
        # Check if filtering is needed
        needs_filtering = date_criteria or search_params.get('participants')
        
        if needs_filtering:
            if self.use_prefiltering:
                print("Using PRE-FILTERING approach")
                retrieved_docs = self._retrieve_with_prefiltering(date_criteria, search_params)
            else:
                print("Using POST-FILTERING approach")
                # Fix: Add missing method name
                retrieved_docs = self._retrieve_with_postfiltering(date_criteria, search_params)
        else:
            # No filtering needed
            retrieved_docs = self.vector_store.similarity_search(
                search_params['query'],
                k=4,
            )
        
        print(f'Number of chunks retrieved: {len(retrieved_docs)}')
        
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('chat_name', 'Unknown')}\n"
            f"Date Range: {doc.metadata.get('date_range', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        
        return serialized, retrieved_docs

    ### Setup LangGraph components
    
    ### Setting up orchestration of our Steps (retrieval and generation) using LangGraph
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def query_or_respond(self, state: MessagesState):
        
        """ Generate tool call for retrieval or direct respond."""

        retrieve_tool = self.create_retrieve_tool()
        llm_with_tools = self.chat_model.bind_tools([retrieve_tool])

        system_msg = SystemMessage(
        "You have access to WhatsApp chat history. Use the retrieve tool only for questions about past conversations, messages, or personal interactions. For general knowledge, respond directly."
        )
        cleaned_messages = [msg for msg in state["messages"] if msg.type != "tool"]
        messages_with_system = [system_msg] + cleaned_messages
        #print(f'Message List: {messages_with_system}')
        
   
        response = llm_with_tools.invoke(messages_with_system)


        return {"messages": [response]}

    def generate(self, state: MessagesState):
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
        response = self.chat_model.invoke(prompt) 
        return {"messages": [response]}

    def initialize_graph_builder(self):
        #tools = ToolNode([self.retrieve])
        retrieve_tool = self.create_retrieve_tool()  
        tools = ToolNode([retrieve_tool])


        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges( 
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"}
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        return graph_builder

    def initialize_graph(self):
        graph_builder = self.initialize_graph_builder()

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        # specify thread_id
        config = {"configurable": {"thread_id": "abc123"}}
        
        # Fix: Return both graph and config
        return graph, config