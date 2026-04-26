import json
import argparse
from pathlib import Path
from corpus_builder import deduplicate_chunks

def main():
    parser = argparse.ArgumentParser(description="Global Chunk Deduplication")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard similarity threshold")
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash permutations")
    args = parser.parse_args()

    # Define paths
    chunks_path = Path("data/chunks/chunks.jsonl")
    if not chunks_path.exists():
        print(f"File not found: {chunks_path}")
        return

    print("=========================================================")
    print("                GLOBAL DEDUPLICATION                     ")
    print("=========================================================")
    print(f"Loading {chunks_path}...")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    
    print(f"Loaded {len(chunks)} total chunks. Inter-topic deduplication started...")
    
    kept_chunks, removed = deduplicate_chunks(
        chunks=chunks, 
        threshold=args.threshold, 
        num_perm=args.num_perm
    )
    
    print(f"Global deduplication complete!")
    print(f"Overlapping chunks removed: {removed}")
    print(f"Unique chunks kept: {len(kept_chunks)}")
    
    # Overwrite with global deduped list
    print("Writing deduplicated chunks...")
    with open(chunks_path, 'w', encoding='utf-8') as out:
        for c in kept_chunks:
            out.write(json.dumps(c) + "\n")
            
    print("Finished.")

if __name__ == '__main__':
    main()