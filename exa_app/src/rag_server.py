import os

# OMP error temporary fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import re
import json
import logging
import unicodedata
import torch
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from deep_translator import GoogleTranslator

# ────────────────────────── 번역기 ──────────────────────────
translator = GoogleTranslator(source='ko', target='en')

# LangChain
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from transformers import BitsAndBytesConfig    


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rag_server.log")]
)



# ────────────────────────── 경로 / 모델 설정 ──────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models")
DATA_DIR        = os.path.join(BASE_DIR, "data")
KO_FILE         = os.path.join(DATA_DIR, "ko_mc.txt")
EN_FILE         = os.path.join(DATA_DIR, "en_mc.txt")

EMB_MODEL_PATH  = os.path.join(BASE_MODEL_PATH, "embeddings",
                               "models--sentence-transformers--all-mpnet-base-v2",
                               "snapshots", "12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")

SLM_MODEL_PATH  = os.path.join(BASE_MODEL_PATH, "slm",
                               "models--LGAI-EXAONE--EXAONE-3.5-2.4B-Instruct",
                               "snapshots", "e949c91dec92095908d34e6b560af77dd0c993f8")

EMB_MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
CHAT_NAME       = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")
EMB_DEVICE = DEVICE_GPU  
# ────────────────────────── 로깅 ──────────────────────────



BASE_DIR   = os.path.dirname(os.path.dirname(__file__))

# ────────────────────────── 캐시 설정 ──────────────────────────
CACHE_DIR = os.path.join(BASE_DIR, "cache")
INDEX_FILE  = os.path.join(CACHE_DIR, "en_index.faiss")
META_FILE   = os.path.join(CACHE_DIR, "meta.json")

logger = logging.getLogger("rag_server")
logger.info(f"[DEBUG] __file__       = {__file__}")
logger.info(f"[DEBUG] cwd            = {os.getcwd()}")
logger.info(f"[DEBUG] BASE_DIR       = {BASE_DIR}")
logger.info(f"[DEBUG] CACHE_DIR      = {CACHE_DIR}")

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
ensure_cache_dir()


def load_data_from_files():
    global ko_blocks, en_blocks, en_index
    if not (os.path.exists(KO_FILE) and os.path.exists(EN_FILE)):
        logger.warning("데이터 파일이 없어 자동 로드를 건너뜁니다.")
        return

    ensure_cache_dir()
    # ➊ 캐시가 있으면 바로 로드
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        logger.info("캐시된 FAISS 인덱스 로드 중…")
        # load blocks 메타
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
            ko_blocks = meta["ko_blocks"]
            en_blocks = meta["en_blocks"]
        # load index
        en_index = faiss.read_index(INDEX_FILE)
        logger.info(f"FAISS index loaded from cache ({en_index.ntotal} vectors).")
        return

    # ➋ 캐시 없으면 새로 로드+임베딩+저장
    logger.info("Loading ko_mc.txt / en_mc.txt …")
    with open(KO_FILE, encoding="utf-8") as fk:
        ko_raw = fk.read()
    with open(EN_FILE, encoding="utf-8") as fe:
        en_raw = fe.read()

    pat = r'(?=[ \t]*"item_name")'
    ko_blocks = [b.strip() for b in re.split(pat, ko_raw) if b.strip()]
    en_blocks = [b.strip() for b in re.split(pat, en_raw) if b.strip()]
    logger.info(f"Korean blocks: {len(ko_blocks)}, English blocks: {len(en_blocks)}")

    # 임베딩 및 인덱스 생성
    vecs = embedder.embed_documents(en_blocks)
    dim  = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vecs, dtype="float32"))
    en_index = index
    logger.info(f"FAISS index built with {en_index.ntotal} vectors.")

    # ➌ 캐시 디스크에 저장
    faiss.write_index(en_index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "ko_blocks": ko_blocks,
            "en_blocks": en_blocks
        }, f, ensure_ascii=False)
    logger.info("Index and metadata saved to cache.")








logger = logging.getLogger("rag_server")
logger.info(f"Using embedding model path: {EMB_MODEL_PATH}")
logger.info(f"Using SLM model path: {SLM_MODEL_PATH}")

# ────────────────────────── FastAPI 앱 ──────────────────────────
app = FastAPI(title="Minecraft RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ────────────────────────── 요청/응답 모델 ──────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ────────────────────────── 유틸 ──────────────────────────
def translate_ko2en(text: str) -> str:
    try:
        return translator.translate(text).strip()
    except Exception as e:
        logger.warning(f"번역 오류: {e}")
        return text

TAG_LIST = [
    "Crafting", "Cooking", "Smelting/Fuel", "Enchanting Resource", "Potion",
    "Diamond Acquisition", "Iron Ore Acquisition", "Gold Ore Acquisition", "Coal Acquisition",
    "Redstone Dust Acquisition", "Lapis Lazuli Acquisition", "Emerald Acquisition",
    "Overworld", "Nether", "End", "Plains", "Desert", "Forest", "Ocean", "Cave",
    "Mountain/Hills", "Underground", "Melee Combat", "Ranged Combat", "Projectile",
    "Mob Drop", "Animal Drop", "Hostile", "Passive", "Farming", "Breeding",
    "Animal Feed", "Fishing", "Bone Meal Production", "Redstone Signal Generator",
    "Redstone Signal Transmission", "Redstone Signal Comparison",
    "Redstone Signal Amplification and Delay", "Automatic Item Collection and Transport",
    "Inventory Management", "Building", "Decoration", "Light Source",
    "Teleportation", "Villager Trading"
]

# ────────────────────────── 임베딩 래퍼 ──────────────────────────
class BgeEmbeddings(Embeddings):
    def __init__(self, model_path=EMB_MODEL_PATH, device=DEVICE_GPU,
                 batch_size=16, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counted

    def _embed(self, texts: List[str]):
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs).astype("float32") if all_embs else np.zeros((0, 768), dtype="float32")

    def embed_documents(self, texts): return self._embed(texts)
    def embed_query(self, text):     return self._embed([text])[0]

# ────────────────────────── LLM 로드 ──────────────────────────
# def load_llm():
#     logger.info("Loading tokenizer…")
#     tok = AutoTokenizer.from_pretrained(SLM_MODEL_PATH, local_files_only=True,
#                                         trust_remote_code=True, use_fast=False)
#     logger.info("Loading EXAONE model…")
#     mdl = AutoModelForCausalLM.from_pretrained(
#         SLM_MODEL_PATH, local_files_only=True, trust_remote_code=True,
#         low_cpu_mem_usage=True
#     )
#     logger.info("LLM loaded.")
#     return tok, mdl

def load_llm():
    logger.info("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(
        SLM_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False
    )

    # 4-bit 양자화 설정 (bnb)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,                       # 4-bit
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16    # (FP16 > BF16 권장)
    )

    logger.info("Loading EXAONE model (4-bit GPU)…")
    mdl = AutoModelForCausalLM.from_pretrained(
        SLM_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    logger.info("LLM loaded (4-bit). BitsAndBytes sets device automatically.")
    return tok, mdl



def generate_text(prompt: str, tok_chat, mdl_chat):
    device  = next(mdl_chat.parameters()).device
    inputs  = tok_chat(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = mdl_chat.generate(
            input_ids      = inputs.input_ids,
            #attention_mask = inputs.attention_mask,
            max_new_tokens = 1024,
            min_new_tokens = 120,   # 최소 길이 강제
            do_sample      = False, # 스트리밍과 동일
            #repetition_penalty = 1.0
        )

    return tok_chat.decode(
        ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

# ────────────────────────── 전역 상태 ─────────────────────────
embedder_cpu = embedder_gpu = tok_chat = mdl_chat = None
ko_blocks: List[str] = []
en_blocks: List[str] = []
en_index  = None

# ────────────────────────── 데이터 자동 로드 ──────────────────────────
def load_data_from_files():
    global ko_blocks, en_blocks, en_index, embedder_cpu
    if not (os.path.exists(KO_FILE) and os.path.exists(EN_FILE)):
        logger.warning("데이터 파일이 없어 자동 로드를 건너뜁니다.")
        return

    logger.info("Loading ko_mc.txt / en_mc.txt …")
    with open(KO_FILE, encoding="utf-8") as fk:
        ko_raw = fk.read()
    with open(EN_FILE, encoding="utf-8") as fe:
        en_raw = fe.read()

    pat = r'(?=[ \t]*"item_name")'
    ko_blocks = [b.strip() for b in re.split(pat, ko_raw) if b.strip()]
    en_blocks = [b.strip() for b in re.split(pat, en_raw) if b.strip()]
    logger.info(f"Korean blocks: {len(ko_blocks)}, English blocks: {len(en_blocks)}")

    vecs = embedder.embed_documents(en_blocks)   # ✅
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vecs, dtype="float32"))
    en_index = index                           # wrapper 대신 raw faiss 객체 저장
    logger.info(f"FAISS index built with {en_index.ntotal} vectors.")

# ────────────────────────── FastAPI 라이프사이클 ──────────────────────────
@app.on_event("startup")
async def _startup():
    global embedder, tok_chat, mdl_chat
    logger.info("Startup: loading models…")
    embedder  = BgeEmbeddings(device=EMB_DEVICE)  
    tok_chat, mdl_chat = load_llm()
    load_data_from_files()
    logger.info("Startup complete.")

# ────────────────────────── API 엔드포인트 ──────────────────────────
@app.get("/")
async def root(): return {"message": "Minecraft RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    global en_index, ko_blocks, en_blocks
    if en_index is None:
        return QueryResponse(answer="참고 데이터가 없습니다. 마인크래프트 관련 질문에 대해 도와드리겠습니다.")

    q_ko = req.query
    q_en = translate_ko2en(q_ko)
    logger.info(f"Query: {q_ko} → {q_en}")

    logger.info("Available TAG_LIST (%d items): %s", len(TAG_LIST), TAG_LIST)
    # 태그 선택 프롬프트 (CPU 추론)
    tag_prompt = (
        "Below is a list of possible Minecraft-related tags. "
        "Choose **1 to 4** tags most relevant to the question, and output them as a comma-separated list (no explanation):\n\n"
        f"Question (English): {q_en}\n\n"
        "Available tags:\n" + ", ".join(TAG_LIST) + "\n\n"
        "Answer:"
    )


    # ────────────────────────── 태그 생성 ──────────────────────────
    tag_inputs = tok_chat(tag_prompt, return_tensors="pt").to(DEVICE_GPU)
    with torch.no_grad():
        tag_out_ids = mdl_chat.generate(
            **tag_inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
        )

    # prompt 토큰 개수만큼 슬라이싱
    prompt_len = tag_inputs.input_ids.shape[1]
    raw_tags = tok_chat.decode(
        tag_out_ids[0, prompt_len:],   # <-- 여기만 디코딩
        skip_special_tokens=True
    ).strip()





    tags = [t.strip() for t in raw_tags.split(",") if t.strip() in TAG_LIST][:4]
    
    logger.info("Selected TAGS: %s", tags)
    q_full = q_en + (" | tags: " + ", ".join(tags) if tags else "")
    logger.info("Using for FAISS search: %s", q_full)

    # 검색
    q_vec = embedder.embed_query(q_full).reshape(1, -1)
    D, I = en_index.search(np.array(q_vec, dtype="float32"),
                           k=min(50, en_index.ntotal))
    
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        snippet = ko_blocks[idx][:80].replace("\n", " ")  # 1줄 미리보기
        logger.info(f"[{rank:02}] score={score:.4f} idx={idx} | {snippet}…")




    ctx_blks = [ko_blocks[i] for i in I[0][:3] if i < len(ko_blocks)]
    if not ctx_blks:
        return QueryResponse(answer="주어진 질문에 대한 정보를 찾을 수 없습니다.",
                             metadata={"matches": 0, "tags": tags})




    prompt = (
        "당신은 마인크래프트 아이템 전문 AI입니다.\n"
        "아래 CONTEXT 내용만 활용해 답하세요.\n"
        "[CONTEXT]\n" + "\n\n".join(ctx_blks) + "\n\n"
        "[QUESTION]\n" + q_ko + "\n"
        "[ANSWER]\n"           # ← 꼭 개행 추가!
    )
    logger.info("\n──────── FINAL PROMPT ────────\n%s\n─────────────────────────────", prompt)

    answer = generate_text(prompt, tok_chat, mdl_chat)



    return QueryResponse(answer=answer, context="\n\n".join(ctx_blks),
                         metadata={"matches": len(ctx_blks), "tags": tags})

# 수동 데이터 로드 엔드포인트 (원본 유지)
@app.post("/load_data")
async def load_data_api(ko_data: str = Body(..., embed=True),
                        en_data: str = Body(..., embed=True)):
    # 1) embedder_cpu 같은 전역은 더 이상 필요 없습니다
    global ko_blocks, en_blocks, en_index
    ensure_cache_dir()

    # 2) 블록 추출 로직 그대로
    pat = r'(?=[ \t]*"item_name")'
    ko_blocks = [b.strip() for b in re.split(pat, ko_data) if b.strip()]
    en_blocks = [b.strip() for b in re.split(pat, en_data) if b.strip()]

    # 3) **CPU 임베더 하나만 사용**
    vecs = embedder.embed_documents(en_blocks)          # ✅

    # 4) **raw-FAISS 인덱스 재생성** (load_data_from_files()와 동일)
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vecs, dtype="float32"))
    en_index = index           
                             # ✅
    faiss.write_index(en_index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "ko_blocks": ko_blocks,
            "en_blocks": en_blocks
        }, f, ensure_ascii=False)

    return {
        "message": "Data loaded",
        "ko_blocks": len(ko_blocks),
        "en_blocks": len(en_blocks),
        "vectors": en_index.ntotal                      # 확인용
    }
# ────────────────────────── 실행 스크립트 ──────────────────────────
if __name__ == "__main__":
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.getenv("RAG_PORT", 8504)))
    args = p.parse_args()
    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
