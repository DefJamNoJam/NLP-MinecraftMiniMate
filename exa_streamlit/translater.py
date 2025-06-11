import re
from pathlib import Path
from deep_translator import GoogleTranslator
from tqdm import tqdm

# 1) 원본 파일 읽기
input_path = Path("mc_data.txt")
raw_text    = input_path.read_text(encoding="utf-8")

# 2) 빈 줄(줄바꿈+공백 허용+줄바꿈) 기준으로 블록 분리
blocks      = re.split(r'(?:\r?\n[ \t]*\r?\n)+', raw_text)

# 3) 동기 번역기 설정
translator  = GoogleTranslator(source='ko', target='en')

# 4~6) 번역 → 즉시 파일에 쓰기
output_path = Path("mc_data_en.txt")
# 먼저 파일을 초기화
output_path.write_text("", encoding="utf-8")

with output_path.open("a", encoding="utf-8") as fout:
    for blk in tqdm(blocks, desc="Translating & saving"):
        if not blk.strip():
            continue
        # 블록 내 각 라인 번역
        translated_lines = []
        for line in blk.splitlines():
            try:
                tr = translator.translate(line)
            except Exception:
                tr = None
            translated_lines.append(tr if isinstance(tr, str) else line)
        # 번역된 블록 완성
        translated_blk = "\n".join(translated_lines)
        # 파일에 즉시 쓰기 (원본 구조처럼 블록 뒤에 빈 줄 추가)
        fout.write(translated_blk)
        fout.write("\n\n")
        fout.flush()

print(f"✅ Done! Translated and saved to {output_path}")
