import argparse
from data_collator import DataCollator  

def main():
    parser = argparse.ArgumentParser(description="Process WhatsApp and Instagram data.")
    parser.add_argument(
        '--source',
        type=str,
        choices=['whatsapp', 'instagram', 'both'],
        default='whatsapp',
        help="Source of data to process: 'whatsapp', 'instagram', or 'both'. Default is 'whatsapp'."
    )
    parser.add_argument(
        '--data_dir_path',
        type=str,
        default='test-data',
        help="Path to the data directory."
    )

    parser.add_argument(
        '--user_name',
        type=str,
        default=None,
        help="Optional user name to override detected user name."
    )

    parser.add_argument(
        '--embed_model',
        type=str,
        default='dengcao/Qwen3-Embedding-0.6B:Q8_0',
        help="Ollama registerd embedding model"
    )
    args = parser.parse_args()
    print(args)
    print(f"Running DataCollator with source='{args.source}' and user_name='{args.user_name}'")
    DataCollator(source=args.source, user_name=args.user_name,embedding_model = args.embed_model,data_dir_path = args.data_dir_path)

if __name__ == "__main__":
    main()