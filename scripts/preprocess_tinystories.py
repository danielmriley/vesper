import sentencepiece as spm
import os
import numpy as np
import argparse

def train_tokenizer(input_file, model_prefix, vocab_size):
    print(f"Training SentencePiece model on {input_file}...")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        num_threads=os.cpu_count()
    )
    print("Tokenizer training complete.")

def encode_file(input_file, model_file, output_file):
    print(f"Encoding {input_file}...")
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    ids = sp.encode(text)
    print(f"Encoded {len(ids)} tokens.")
    
    # Save as uint16 (since vocab < 65536)
    ids_array = np.array(ids, dtype=np.uint16)
    with open(output_file, 'wb') as f:
        f.write(ids_array.tobytes())
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input text file')
    parser.add_argument('--output_dir', type=str, default='data_bin', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    model_prefix = os.path.join(args.output_dir, 'tokenizer')
    output_bin = os.path.join(args.output_dir, 'train.bin')
    
    # Train tokenizer
    train_tokenizer(args.input, model_prefix, args.vocab_size)
    
    # Encode
    encode_file(args.input, model_prefix + '.model', output_bin)

if __name__ == '__main__':
    main()
