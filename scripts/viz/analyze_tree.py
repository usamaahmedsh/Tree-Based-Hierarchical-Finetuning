import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_tree():
    leaf_path = Path("data/tree/leaf_nodes.json")
    if not leaf_path.exists():
        print("Leaf nodes file not found")
        return

    with open(leaf_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_chunks = 0
    total_leaves = len(data)
    
    # Layer counters
    topics = Counter()
    emotions = Counter()
    demographics = Counter()
    registers = Counter()
    readabilities = Counter()

    leaf_sizes = []

    for leaf_id, leaf_info in data.items():
        count = leaf_info.get("document_count", 0)
        total_chunks += count
        leaf_sizes.append((leaf_id, count, leaf_info.get("label_path", "")))
        
        labels = leaf_info.get("layer_labels", {})
        topics[labels.get("topic", "Unknown")] += count
        emotions[labels.get("emotion", "Unknown")] += count
        demographics[labels.get("demographic", "Unknown")] += count
        registers[labels.get("register", "Unknown")] += count
        readabilities[labels.get("readability", "Unknown")] += count

    print("="*50)
    print("TREE ANALYSIS REPORT")
    print("="*50)
    print(f"Total Chunks: {total_chunks}")
    print(f"Total Leaf Nodes: {total_leaves}")
    if total_leaves > 0:
        print(f"Average Chunks per Leaf: {total_chunks / total_leaves:.2f}")

    leaf_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- TOP 5 LARGEST LEAVES ---")
    for i in range(min(5, len(leaf_sizes))):
        print(f"{i+1}. {leaf_sizes[i][2]} (ID: {leaf_sizes[i][0]}) -> {leaf_sizes[i][1]} chunks")
        
    print("\n--- TOP 3 TOPICS ---")
    for t, c in topics.most_common(3):
        print(f"{t}: {c} chunks ({c/total_chunks*100:.1f}%)")
        
    print("\n--- TOP 3 EMOTIONS ---")
    for e, c in emotions.most_common(3):
        print(f"{e}: {c} chunks ({c/total_chunks*100:.1f}%)")

    print("\n--- TOP 3 DEMOGRAPHICS ---")
    for d, c in demographics.most_common(3):
        print(f"{d}: {c} chunks ({c/total_chunks*100:.1f}%)")

    print("\n--- TOP 3 REGISTERS/FORMALITY ---")
    for r, c in registers.most_common(3):
        print(f"{r}: {c} chunks ({c/total_chunks*100:.1f}%)")
        
    print("\n--- TOP 3 READABILITY LEVELS ---")
    for r, c in readabilities.most_common(3):
        print(f"{r}: {c} chunks ({c/total_chunks*100:.1f}%)")
        
    print("\n--- LAYER DISTRIBUTION SUMMARY ---")
    print(f"Unique Topics (Layer 1):       {len(topics)}")
    print(f"Unique Emotions (Layer 2):     {len(emotions)}")
    print(f"Unique Demographics (Layer 3): {len(demographics)}")
    print(f"Unique Registers (Layer 4):    {len(registers)}")
    print(f"Unique Readability (Layer 5):  {len(readabilities)}")

if __name__ == "__main__":
    analyze_tree()
