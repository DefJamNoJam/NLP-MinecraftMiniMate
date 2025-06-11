import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import json
import hashlib
from pathlib import Path
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# ────────────────────────── 설정 ──────────────────────────
EMB_MODEL_NAME = "jhgan/ko-sbert-nli"
batch_size = 16  # 배치 단위로 임베딩하여 OOM 방지
device = torch.device("cpu")  # 테스트는 CPU로 실행하여 메모리 부족 방지

# ────────────────────────── BGE 임베딩 래퍼 ──────────────────────────
class BgeEmbeddings:
    def __init__(self, model_name=EMB_MODEL_NAME, device=device, batch_size=batch_size, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
    
    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counted

    def embed(self, texts):
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            emb = self._mean_pool(out.last_hidden_state, inputs.attention_mask)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype="float32")
        return np.vstack(all_embs).astype("float32")

# ────────────────────────── 블록 분리 함수 ──────────────────────────
def load_blocks(path: Path):
    raw = path.read_text(encoding="utf-8")
    pat = r'(?:\r?\n[ \t]*\r?\n)+'
    blocks = [b.strip() for b in re.split(pat, raw) if b.strip()]
    return blocks

# ────────────────────────── 테스트 스크립트 ──────────────────────────
if __name__ == "__main__":
    # 파일 경로 설정
    ko_path = Path("mc_data.txt")
    en_path = Path("mc_data_en.txt")
    if not ko_path.exists() or not en_path.exists():
        print("ERROR: mc_data.txt 또는 mc_data_en.txt 파일이 필요합니다.")
        exit(1)

    # 블록 로드
    ko_blocks = load_blocks(ko_path)
    en_blocks = load_blocks(en_path)
    print(f"한글 블록: {len(ko_blocks)}, 영어 블록: {len(en_blocks)}")

    # 인덱스 생성
    embedder = BgeEmbeddings()
    print("영어 블록 임베딩 중… (CPU 배치 처리)")
    vecs = embedder.embed(en_blocks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    print("FAISS 인덱스 생성 완료.")

    # 테스트 쿼리: 첫 한글 블록 사용
    query_text = ko_blocks[0]
    print("\n== 테스트 쿼리 한글 블록 ==")
    print(query_text[:200].replace("\n"," ") + "…")

    # 한글 쿼리 임베딩 및 검색
    print("영어로 번역 없이 직접 검색 (샘플)")
    q_emb = embedder.embed([query_text])
    D, I = index.search(q_emb, k=min(10, index.ntotal))

    # 유효 인덱스만 선별
    en_pairs = []  # (orig_idx, block, score)
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(en_blocks):
            en_pairs.append((idx, en_blocks[idx], score))
    print("\n== 매칭된 영어 후보 ==")
    if not en_pairs:
        print("(검색 결과가 없습니다.)")
    else:
        for idx, txt, score in en_pairs:
            snippet = txt[:80].replace("\n", " ")
            print(f"인덱스 {idx}, score={score:.4f}, snippet={snippet}…")

    # 영어 후보 매핑된 한글 블록
    print("\n== 영어 후보 매핑된 한글 블록 ==")
    for orig_idx, txt, score in en_pairs:
        if orig_idx < len(ko_blocks):
            print(f"영어 idx {orig_idx} -> 한글 idx {orig_idx}")
            print(ko_blocks[orig_idx][:80].replace("\n"," ") + "…\n")

    print("테스트 완료.")
