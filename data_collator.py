# base
import os
import zipfile
import re
import shutil
import json
import ast
from datetime import datetime

# third-party
import pandas as pd
import chromadb

# rag_chatbot
from building_rag.vectorize_files import vectorize_chats

class DataCollator:
    """
    Entry point that orchestrates WhatsApp and Instagram data collation and vectorization.
    """

    def __init__(self, source='whatsapp', user_name=None,embedding_model=None,data_dir_path = None):
        """
        :param source: str, one of ['whatsapp', 'instagram', 'both']
        :param user_name: str, optional override for detected user name
        """
        self.source = source
        self.user_name = user_name
        self.embedding_model = embedding_model
        self.data_dir_path = data_dir_path

        if source == 'whatsapp':
            self._process_whatsapp()
        elif source == 'instagram':
            self._process_instagram()
        elif source == 'both':
            self._process_both()

        else:
            raise ValueError("Source must be one of ['whatsapp', 'instagram', 'both']")

    def _process_whatsapp(self):
        print("\n--- Processing WhatsApp data ---")
        whatsapp_collator = WhatsAppDataCollator(user_name=self.user_name,data_dir_path=self.data_dir_path)
        chat_path = os.path.join(whatsapp_collator.save_dir,'whatsapp_chats.csv')
        if os.path.exists(chat_path):
            vectorize_chats(chat_path=chat_path, source='whatsapp',embedding_model_name=self.embedding_model)
        else:
            print(f"WhatsApp chat file not found at {chat_path}, skipping vectorization.")

    def _process_instagram(self):
        print("\n--- Processing Instagram data ---")
        instagram_collator = InstagramDataCollator(user_name=self.user_name,data_dir_path=self.data_dir_path)
        chat_path = os.path.join(instagram_collator.save_dir, 'instagram_chats.csv')
        if os.path.exists(chat_path):
            vectorize_chats(chat_path=chat_path, source = 'instagram',embedding_model_name=self.embedding_model)
        else:
            print(f"Instagram chat file not found at {chat_path}, skipping vectorization.")

    def _process_both(self):
        print("\n--- Processing Instagram data ---")
        instagram_collator = InstagramDataCollator(user_name=self.user_name)
        ig_chat_path = os.path.join(instagram_collator.save_dir, 'instagram_chats.csv')

        whatsapp_collator = WhatsAppDataCollator(user_name=self.user_name)
        wa_chat_path = os.path.join(whatsapp_collator.save_dir,'whatsapp_chats.csv')
        
        print("\n--- Processing WhatsApp data ---")
        if os.path.exists(wa_chat_path) and os.path.exists(ig_chat_path):
            vectorize_chats(chat_path=wa_chat_path, chat_path2=ig_chat_path, source = 'instagram',embedding_model_name=self.embedding_model)
        else:
            print(f"Instagram chat file not found at {ig_chat_path}, or WhatsApp chat file not found at {wa_chat_path}, skipping vectorization..")


    def detect_user_name(self, df: pd.DataFrame, chat_col="chat_id", participants_col="participants"):
        """
        Detect the most likely user name (yourself) by finding the participant 
        who appears in the largest number of distinct chats.

        Optimized version: uses explode + groupby instead of manual loops.
        """

        if df.empty or chat_col not in df.columns or participants_col not in df.columns:
            print("No data or required columns missing.")
            return None

        # Work on distinct chats only
        chat_participants = df[[chat_col, participants_col]].drop_duplicates(subset=[chat_col])

        # Parse participants col into lists (only once per chat, not per row)
        if chat_participants[participants_col].apply(lambda x: isinstance(x, str)).any():
            chat_participants[participants_col] = chat_participants[participants_col].apply(ast.literal_eval)

        # Explode so each row = (chat_id, participant)
        exploded = chat_participants.explode(participants_col)

        # Count in how many chats each participant appears
        participant_counts = (
            exploded.groupby(participants_col)[chat_col]
            .nunique()               # number of distinct chats per participant
            .sort_values(ascending=False)
        )

        most_common_user = participant_counts.index[0]
        num_chats = participant_counts.iloc[0]

        print(f"Auto-detected user name: {most_common_user} (appears in {num_chats} chats)")
        return most_common_user

class WhatsAppDataCollator(DataCollator):
    """
    In WhatsApp, messages are downloaded per invidual, and they are zipped.
    """

    def __init__(self,user_name = None, data_dir_path= 'test_data'):
        self.source_dir = data_dir_path + '/whatsapp/zipped_chats'
        self.output_dir =  data_dir_path + '/whatsapp//whatsapp_chats'
        self.save_dir = data_dir_path + '/processed_chats'
        self.user_name = user_name

        self.create_directories()

        if self.should_extract_chats():
            self.extract_data()
        else:
            print('Processed chats  directory exists and populated, no extraction required.')

    def should_extract_chats(self):
        csv_path = self.save_dir + '/whatsapp_chats.csv'
        return not os.path.exists(csv_path)

    def create_directories(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f'Created/verified WhatsApp chats directory: {self.output_dir}')
        except Exception as e:
            print(f'Error creating WhatsApp chats directory: {e}')
        
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f'Created/verified processed chats directory: {self.save_dir}')
        except Exception as e:
            print(f'Error creating processed chats directory: {e}')     

    def extract_data(self):
        self.unzip_chats()
        df = self.parse_chat_history()
        if self.user_name is None:
            self.user_name = self.detect_user_name(df)

        df = self.extract_structured_info(df)

        df.to_csv(self.save_dir + '/whatsapp_chats.csv')

    def unzip_chats(self):
        try:
            for file_name in os.listdir(self.source_dir):
                if file_name.endswith('zip'):
                    file_path = self.source_dir + '/' + file_name
                    try:
                        with zipfile.ZipFile(file_path,'r') as zip_object:
                            for file_info in zip_object.filelist:
                                file_info.filename = file_name.replace('.zip','.txt')
                                try:
                                    zip_object.extract(file_info,self.output_dir)
                                except Exception as e:
                                    print(f'Error extracting file {file_info.filename}: {e}')
                    except Exception as e:
                        print(f'Error processing {file_name}: {e}')
        except FileNotFoundError as e:
            print(f'Error: Source directory not found: {e}')
        except Exception as e:
            print(f'Unexpected error in unzip_chats: {e}')


    # def parse_message(self,line):
    #     """Extract from message specfic content"""
    #     pattern = r'\[(\d{2}\.\d{2}\.\d{4}), (\d{2}:\d{2}:\d{2})\] ~? ?([^:]*(?::[^: ])*[^:]*): (.*)'
        

    #     match = re.match(pattern, line)
        
    #     if match:
    #         return {'date' : match.group(1),
    #                 'time': match.group(2),
    #                 'sender': match.group(3),
    #                 'content': match.group(4)}
    #     return None

    def parse_message(self, line):
        """
        Parse a single WhatsApp message line.

        Supports both formats:
        1️⃣ [DD.MM.YYYY, HH:MM(:SS)] Sender: Message
        2️⃣ D/M/YY, HH:MM - Sender: Message
        Also handles system messages with no sender.
        """

        # --- Format 1: [04.04.2025, 19:59:00] Name: Text ---
        pattern_bracketed = r'^\[(\d{1,2}\.\d{1,2}\.\d{2,4}), (\d{1,2}:\d{2}(?::\d{2})?)\]\s([^:]+):\s?(.*)$'
        # --- Format 2: 4/4/25, 19:59 - Name: Text ---
        pattern_plain = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - ([^:]+): (.*)$'
        # --- Format 2 (system messages, no sender): 4/4/25, 19:59 - Message ---
        pattern_plain_system = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*)$'

        # Try bracketed format first
        match = re.match(pattern_bracketed, line)
        if match:
            return {
                'date': match.group(1),
                'time': match.group(2),
                'sender': match.group(3).strip(),
                'content': match.group(4).strip(),
            }

        # Try plain format (with sender)
        match = re.match(pattern_plain, line)
        if match:
            return {
                'date': match.group(1),
                'time': match.group(2),
                'sender': match.group(3).strip(),
                'content': match.group(4).strip(),
            }

        # Try plain format (system message)
        match = re.match(pattern_plain_system, line)
        if match:
            return {
                'date': match.group(1),
                'time': match.group(2),
                'sender': None,
                'content': match.group(3).strip(),
            }

        # Nothing matched
        return None




    # def split_chat(self,chat_text):
    #     """Split WhatsApp chat by message timestamps, preserving multiline messages"""
    #     # Pattern to match the timestamp at start of message
    #     pattern = r'(?=\[\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2}:\d{2}\])'
    #     # Split by the pattern and filter out empty strings
    #     messages = [msg.strip() for msg in re.split(pattern, chat_text) if msg.strip()]
        
    #     return messages

    def split_chat(self, chat_text):
        """
        Split WhatsApp chat into individual messages by timestamp,
        supporting both bracketed and plain date/time formats.
        """
        # Pattern 1: [DD.MM.YYYY, HH:MM(:SS)]
        bracketed_pattern = r'(?=\[\d{1,2}\.\d{1,2}\.\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?\])'

        # Pattern 2: D/M/YY, HH:MM -
        plain_pattern = r'(?=\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - )'

        # Combine both using alternation
        combined_pattern = f'{bracketed_pattern}|{plain_pattern}'

        # Split and clean up
        messages = [msg.strip() for msg in re.split(combined_pattern, chat_text) if msg.strip()]

        return messages

    def parse_chat(self,chat,chat_id,chat_name):
        chat_lines = self.split_chat(chat)
        message_dicts = []

        for line in chat_lines:
            message_dict = self.parse_message(line)
            if message_dict:
                message_dict['chat_id'] = chat_id
                message_dict['chat_name'] = chat_name
                message_dicts.append(message_dict)
        return message_dicts

    def parse_chat_history(self):
        all_messages = []
        chat_id = 0

        for file_name in os.listdir(self.output_dir):
            if file_name.lower().startswith('whatsapp'):

                file_path = self.output_dir + '/' + file_name
                with open(file_path,'r') as file:
                    text = file.read()
                    print(f"File path : {file_path}")
                    chat_name = file_name.replace('WhatsApp Chat - ','').replace('WhatsApp Chat with','').replace('.txt','')
                    print
                    chat_messages = self.parse_chat(text,chat_id,chat_name)            
                    print(chat_messages)
                    chat_id+=1

                    all_messages.extend(chat_messages)
        chat_df = pd.DataFrame(all_messages)

        return chat_df
    
    def extract_structured_info(self,df):
        def remove_gc_participant(row):
            """ Groupchat name mistakenly added as first paritipant, manually remove after. """
            if row['is_groupchat']:
                return row['participants'][1:]
            else:
                return row['participants']
            
        def process_chat_name(row):
            """ Make sure chat name corresponds to name of sender. """
            if row['is_groupchat']:
                return row['chat_name']
            else:
                names = [name for name in row['participants'] if name!='Dan']
                return names[0]
    
        df['is_groupchat'] = df['chat_id'].map(df.groupby('chat_id')['sender'].nunique()>2) 
        df['participants'] = df['participants'] = df.groupby('chat_id')['sender'].transform(lambda x: [list(filter(None, x.unique()))] * len(x))

        df['participants'] = df.apply(remove_gc_participant, axis=1)
        df['chat_name'] = df.apply(process_chat_name, axis = 1)

        # make me main user
        df.loc[df['sender'] == self.user_name, 'sender'] = 'Me'
        
        return df


class InstagramDataCollator(DataCollator):
    """
    Instagram Data Collator.
    Instagram data is obtained by requesting it through the App (Download your information).
    """
    
    def __init__(self, user_name=None, data_dir_path= 'test_data'):
        self.user_name = user_name
        self.source_dir = data_dir_path + '/instagram/instagram_data'
        self.save_dir = data_dir_path + '/processed_chats'
        self.inbox_path = None
        
        self.create_directories()
        self.find_inbox_path()
        self.extract_data()


    def create_directories(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f'Created/verified processed chats directory: {self.save_dir}')
        except Exception as e:
            print(f'Error creating processed chats directory: {e}')
    
    def find_inbox_path(self):
        """Find the inbox path within the Instagram data structure"""
        try:
            for root, dirs, files in os.walk(self.source_dir):
                if 'inbox' in dirs and 'messages' in root:
                    self.inbox_path = os.path.join(root, 'inbox')
                    print(f'Found inbox at: {self.inbox_path}')
                    return
            raise FileNotFoundError("Could not find inbox folder in Instagram data")
        except Exception as e:
            print(f'Error finding inbox path: {e}')
            self.inbox_path = None
    
    def cleanup_media_files(self):
        """Remove photos and videos folders from chat directories"""
        if not self.inbox_path:
            return
            
        try:
            for item_name in os.listdir(self.inbox_path):
                item_path = os.path.join(self.inbox_path, item_name)
                if os.path.isdir(item_path):
                    try:
                        for sub_item in os.listdir(item_path):
                            sub_path = os.path.join(item_path, sub_item)
                            if os.path.isdir(sub_path) and sub_item.lower() in ['photos', 'videos']:
                                shutil.rmtree(sub_path)
                                print(f"Deleted folder: {sub_path}")
                    except Exception as e:
                        print(f"Error accessing {item_path}: {e}")
            print("Cleanup completed.")
        except Exception as e:
            print(f'Error during cleanup: {e}')
    
    def fix_encoding(self, text):
        """Fix encoding issues in Instagram data"""
        if isinstance(text, str):
            try:
                return text.encode('latin1').decode('utf-8')
            except:
                return text
        return text
    
    def process_messages(self):
        """Process Instagram message files into a pandas DataFrame"""
        if not self.inbox_path:
            return pd.DataFrame()
            
        all_messages = []
        try:
            for chat_folder in os.listdir(self.inbox_path):
                chat_path = os.path.join(self.inbox_path, chat_folder)
                if not os.path.isdir(chat_path):
                    continue
                
                message_files = [f for f in os.listdir(chat_path) if f.startswith('message_') and f.endswith('.json')]
                
                for message_file in message_files:
                    file_path = os.path.join(chat_path, message_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        chat_name = data.get('title', chat_folder)
                        chat_id = chat_folder
                        
                        for msg in data.get('messages', []):
                            timestamp = datetime.fromtimestamp(msg['timestamp_ms'] / 1000)
                            message_data = {
                                'message': msg.get('content', ''),
                                'sender': msg.get('sender_name', ''),
                                'time': timestamp,
                                'chat_id': chat_id,
                                'chat_name': chat_name
                            }
                            all_messages.append(message_data)
                    except Exception as e:
                        print(f'Error processing {message_file}: {e}')
            
            df = pd.DataFrame(all_messages)
            if not df.empty:
                df = df.sort_values('time').reset_index(drop=True)
                
                # Fix encoding issues
                df['sender'] = df['sender'].apply(self.fix_encoding)
                df['chat_name'] = df['chat_name'].apply(self.fix_encoding)
                df['message'] = df['message'].apply(self.fix_encoding)
                
                # Filter out empty messages
                df = df[df['message'].notna() & (df['message'].str.strip() != '')]

                # Filter out media messages 
                df = df.loc[~df['message'].str.contains('sent an attachment.', na=False)]

                df['is_groupchat'] = df['chat_id'].map(df.groupby('chat_id')['sender'].nunique()>2) 
                df['participants'] = df.groupby('chat_id')['sender'].transform(lambda x: [list(x.unique())] * len(x))

                # Map int chat_id
                df['chat_id'] = pd.factorize(df['chat_id'])[0] + 1 
                df.rename(columns={'message' : 'content',},inplace=True)

                df['time'] = pd.to_datetime(df['time'])
                date_col = df['time'].dt.date
                time_col = df['time'].dt.time

                # Drop original and add new columns
                df = df.drop('time', axis=1)
                df['date'] = date_col
                df['time'] = time_col
                
                # Make user main sender
                if self.user_name is None:
                    self.user_name = self.detect_user_name(df)

                df.loc[df['sender'] == self.user_name, 'sender'] = 'Me'

            return df
            
        except Exception as e:
            print(f'Error processing messages: {e}')
            return pd.DataFrame()
    
    def extract_data(self):
        """Main extraction method"""
        if not self.inbox_path:
            print("No inbox path found, skipping extraction")
            return
        try:
            self.cleanup_media_files()
            messages_df = self.process_messages()
            
            if not messages_df.empty:
                messages_df.to_csv(self.save_dir + '/instagram_chats.csv', index=False)
            else:
                print("No messages found to process")
                
        except Exception as e:
            print(f'Error in extract_data: {e}')