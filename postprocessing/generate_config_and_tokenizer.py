import argparse
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_dir', type=str, help="model checkpoint folder. Usually checkpoint_XXXX/.")
parser.add_argument("model_name", type=str, help="model name. E.g., gpt2")
parser.add_argument("--tokenizer_name", type=str)
args = parser.parse_args()

if not args.tokenizer_name:
    args.tokenizer_name = args.model_name

model = transformers.AutoModel.from_pretrained(args.model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name)

model.config.save_pretrained(args.checkpoint_dir)
tokenizer.save_pretrained(args.checkpoint_dir)