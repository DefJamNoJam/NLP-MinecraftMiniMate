import re

def split_by_item_name(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 'item_name'을 기준으로 분리 (첫 번째 빈 항목 제외)
    parts = re.split(r'(?="item_name")', text)
    chunks = [p.strip() for p in parts if p.strip()]
    return len(chunks)

file_ko = 'ko_mc.txt'
file_en = 'en_mc.txt'

ko_chunks = split_by_item_name(file_ko)
en_chunks = split_by_item_name(file_en)

print(f"Chunks in {file_ko}: {ko_chunks}")
print(f"Chunks in {file_en}: {en_chunks}")

if ko_chunks == en_chunks:
    print("✅ The number of chunks is the same.")
else:
    print("⚠️ The number of chunks is different.")
