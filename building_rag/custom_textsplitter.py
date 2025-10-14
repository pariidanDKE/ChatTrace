from typing import List, Any
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from datetime import datetime
import pandas as pd
import ast

class CustomTextSplitter(TextSplitter):
    """Custom Splitter for different Chat Sources"""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    
    def split_text(self, text: str) -> List[str]:
        """Use parent TextSplitter's split_text implementation"""
        return super().split_text(text)
    
    def _chunk_messages(self, full_df: pd.DataFrame, chunk_size: int = 5000) -> List[dict]:
        """Split messages into chunks based on size"""
        chunks = []
        for chat_id in full_df['chat_id'].unique():
            df = full_df[full_df['chat_id'] == chat_id].copy()
            df = df.sort_values('date')  # Ensure chronological order
            
            base_metadata = {
                'chat_id': chat_id,
                'chat_name': df['chat_name'].iloc[0],
                'is_groupchat': df['is_groupchat'].iloc[0],
                #'chat_language': df['chat_language'].iloc[0],
                'participants': df['participants'].iloc[0],
            }
            
            current_chunk = ""
            current_messages = []
            
            for _, row in df.iterrows():
                message = f"{row['sender']}: {row['content']}\n"
                
                # If adding this message exceeds chunk_size, finalize current chunk
                if len(current_chunk + message) > chunk_size and current_chunk:
                    chunks.append({
                        'chunk_text': current_chunk.strip(),
                        'date_range': f"{current_messages[0]['date']} - {current_messages[-1]['date']}",
                        'chunk_type': 'size_based',
                        'part': len([c for c in chunks if c['chat_id'] == chat_id]) + 1,
                        'messages_count': len(current_messages),
                        **base_metadata
                    })
                    current_chunk = ""
                    current_messages = []
                
                current_chunk += message
                current_messages.append(row.to_dict())
            
            # Add final chunk
            if current_chunk:
                chunks.append({
                    'chunk_text': current_chunk.strip(),
                    'date_range': f"{current_messages[0]['date']} - {current_messages[-1]['date']}",
                    'chunk_type': 'size_based',
                    'part': len([c for c in chunks if c['chat_id'] == chat_id]) + 1,
                    'messages_count': len(current_messages),
                    **base_metadata
                })
        chunk_df = pd.DataFrame(chunks)
        #chunk_df.drop(columns=['chat_id'],inplace=True)
        chunk_df.drop_duplicates(inplace=True)

        return chunk_df
    

    def parse_date_flexible(self, date_str):
        """
        Parse date string supporting multiple formats:
        - DD.MM.YYYY
        - YYYY-MM-DD
        - D/M/YY or D/M/YYYY
        - M/D/YY or M/D/YYYY
        """
        from datetime import datetime

        date_str = date_str.strip()

        # Try these formats in order — WhatsApp often uses short slash dates
        possible_formats = [
            "%d.%m.%Y",  # 04.04.2025
            "%Y-%m-%d",  # 2025-04-04
            "%d/%m/%y",  # 4/4/25
            "%m/%d/%y",  # 4/4/25 (US-style)
            "%d/%m/%Y",  # 4/4/2025
            "%m/%d/%Y",  # 4/4/2025 (US-style)
        ]

        for fmt in possible_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If all formats fail, raise an explicit error
        raise ValueError(
            f"Date '{date_str}' doesn't match supported formats: "
            "DD.MM.YYYY, YYYY-MM-DD, D/M/YY, or D/M/YYYY"
        )


    def _create_doc_list(self, df: pd.DataFrame) -> List[Document]:
        """Create Document objects from chunked data"""
        documents = []
        for _, row in df.iterrows():

            metadata_header = f"""**Metadata of Conversation**
        Chat: {row['chat_name']} (part {row['part']}) | Date: {row['date_range']}
        **End of Metadata of Conversation** \n """
            enhanced_chunk_text = metadata_header + row["chunk_text"]
            
            # Parse date range
            start_str, end_str = row['date_range'].split(' - ')
            start_date = int(self.parse_date_flexible(start_str).strftime("%Y%m%d"))
            end_date = int(self.parse_date_flexible(end_str).strftime("%Y%m%d"))
            
            # Determine other person
            is_gc = row['is_groupchat']
            if not is_gc:
                participants = ast.literal_eval(row['participants']) if isinstance(row['participants'], str) else row['participants']
                other_persons = [person for person in participants if person != 'Me']
                other_person = other_persons[0] if other_persons else 'N/A'
            else:
                other_person = 'N/A'
            
            document = Document(
                page_content=enhanced_chunk_text,
                metadata={
                    'date_range': row['date_range'],
                    'start_date': start_date,
                    'end_date': end_date,
                    'part': row['part'],
                    'messages_count': row['messages_count'],
                    'chat_name': row['chat_name'],
                    'is_groupchat': row['is_groupchat'],
                    'chat_id': row['chat_id'],
                    'participants': row['participants'],
                    'other_person': other_person,
                    'source': 'instagram'
                }
            )
            documents.append(document)
        return documents
    
    def split_messages(self, chunk_size: int, message_df: pd.DataFrame, return_df : bool = False) -> List[Document]:
        """Main method to split messages and return Document objects"""

        chunk_df = self._chunk_messages(message_df, chunk_size=chunk_size)
        if return_df:
            return chunk_df

        documents = self._create_doc_list(chunk_df)
        return documents