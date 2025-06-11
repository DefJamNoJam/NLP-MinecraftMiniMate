import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import re
import hashlib
import streamlit as st
import faiss
import shutil
import json
import numpy as np
from deep_translator import GoogleTranslator
import threading
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from langchain.embeddings.base import Embeddings
from langchain.callbacks.base import BaseCallbackHandler

# ────────────────────────── 모델·경로 설정 ──────────────────────────
EMB_MODEL_NAME       = "sentence-transformers/all-mpnet-base-v2"
CHAT_NAME            = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
DEVICE_GPU           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU           = torch.device("cpu")  # 영어 임베딩용 CPU
CACHE_DIR            = ".exa_bge_rerank_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ────────────────────────── 번역 전용 모델 로드 ──────────────────────────
translator = GoogleTranslator(source='ko', target='en')

st.set_page_config(page_title="마인크래프트 RAG (한영 데이터)", layout="wide")
st.title("⛏️ 마인크래프트 아이템 RAG 챗봇 (한영 데이터)")

# ────────────────────────── exa_translate() 대신 쓸 함수 ──────────────────────────
def translate_ko2en(text: str) -> str:
    """
    deep_translator 기반 한→영 번역.
    실패하면 원문 반환.
    """
    try:
        return translator.translate(text).strip()
    except Exception as e:
        st.warning(f"번역 오류: {e}")
        return text




# ────────────────────────── BGE Embeddings 래퍼 (GPU/CPU 지원) ──────────────────────────
class BgeEmbeddings(Embeddings):
    def __init__(self, model_name=EMB_MODEL_NAME, device=DEVICE_GPU, batch_size=16, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        self.model     = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.device    = device
        self.batch_size= batch_size
        self.max_length= max_length

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counted

    def _embed(self, texts):
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
            emb = self._mean_pool(out.last_hidden_state, encoded['attention_mask'])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        if not all_embs:
            return np.zeros((0, self.model.config.hidden_size), dtype='float32')
        return np.vstack(all_embs).astype('float32')

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

# ────────────────────────── 모델 통합 로드 ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm_only():
    # ★ Cross-Encoder 완전 제거
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    tok_c = AutoTokenizer.from_pretrained(CHAT_NAME, trust_remote_code=True)
    mdl_c = AutoModelForCausalLM.from_pretrained(
        CHAT_NAME, trust_remote_code=True,
        quantization_config=bnb, device_map={'':0}
    )
    return tok_c, mdl_c

# ────────────────────────── LLM 스트리밍 헬퍼 ──────────────────────────
def stream_generate(prompt: str):
    llm_device = st.session_state.mdl_chat.device
    inp = st.session_state.tok_chat(prompt, return_tensors="pt").to(llm_device)
    streamer = TextIteratorStreamer(
        st.session_state.tok_chat, skip_prompt=True, skip_special_tokens=True
    )
    thread = threading.Thread(
        target=st.session_state.mdl_chat.generate,
        kwargs={
            "input_ids": inp.input_ids,
            "attention_mask": inp.attention_mask,
            "streamer": streamer,
            "max_new_tokens": 1024,
            "do_sample": False
        },
    )
    thread.start()
    for token in streamer:
        yield token
    thread.join()

# ────────────────────────── StreamHandler 콜백 ──────────────────────────
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response=None, **kwargs):
        self.container.markdown(self.text)

# ────────────────────────── 모델 로드 ──────────────────────────
with st.sidebar.status("모델 로드 중...", expanded=True) as status:
    try:
        status.write(f"임베딩 모델 ({EMB_MODEL_NAME}) 로드 중...")
        st.session_state.embedder_gpu = BgeEmbeddings(device=DEVICE_GPU)
        st.session_state.embedder_cpu = BgeEmbeddings(device=DEVICE_CPU)
        status.write("임베딩 모델 로드 완료.")
        tok_chat, mdl_chat = load_llm_only()
        st.session_state.update({
            'tok_chat': tok_chat,
            'mdl_chat': mdl_chat,
            'models_loaded': True
        })
        status.update(label="모델 로드 완료!", state='complete', expanded=False)
    except Exception as e:
        status.update(label=f"모델 로드 실패: {e}", state='error', expanded=True)
        st.stop()

# ────────────────────────── 사이드바: 한영 파일 업로드 ──────────────────────────
with st.sidebar:
    st.title("데이터 파일 (한영)")
    ko_file = st.file_uploader("한글 TXT 업로드", type=["txt"], key="ko_file")
    en_file = st.file_uploader("영어 TXT 업로드", type=["txt"], key="en_file")
    if st.button("🗑️ 캐시 초기화", key="clear_all"):
        for k in ['en_index','ko_blocks','en_blocks','uploaded_files_key','messages']:
            st.session_state.pop(k, None)
        st.experimental_rerun()

# ────────────────────────── 업로드 파일 처리 & 영어 인덱스 생성 ──────────────────────────
if ko_file and en_file:
    st.sidebar.info("1️⃣ 파일 읽는 중…")
    ko_raw = ko_file.getvalue().decode('utf-8')
    en_raw = en_file.getvalue().decode('utf-8')

    st.sidebar.info("2️⃣ 블록 분할 중…")
    #pat = r'(?:\r?\n[ \t]*\r?\n)+'
    pat = r'(?=[ \t]*"item_name")' 
    ko_blocks = [b.strip() for b in re.split(pat, ko_raw) if b.strip()]
    en_blocks = [b.strip() for b in re.split(pat, en_raw) if b.strip()]

    key = hashlib.md5((ko_raw+en_raw).encode('utf-8')).hexdigest()
    idx_en_path = os.path.join(CACHE_DIR, f"{key}_en.faiss")
    meta_en_path = os.path.join(CACHE_DIR, f"{key}_en.meta")

    # 캐시 로드
    if os.path.exists(idx_en_path) and os.path.exists(meta_en_path):
        st.sidebar.info("🔁 영어 인덱스 캐시 로드 중…")
        idx_en = faiss.read_index(idx_en_path)
        en_blocks = []
        with open(meta_en_path, 'r', encoding='utf-8') as f_meta:
            for line in f_meta:
                line=line.strip()
                if not line: continue
                try: blk=json.loads(line)
                except: blk=line
                en_blocks.append(blk)
        st.sidebar.success("✅ 영어 인덱스 캐시 로드 완료")
    else:
        st.sidebar.info("3️⃣ 영어 블록 임베딩(GPU->CPU 배치) 중…")
        # CPU embed
        vecs_en = st.session_state.embedder_cpu.embed_documents(en_blocks)
        idx_en = faiss.IndexFlatIP(len(vecs_en[0]))
        idx_en.add(np.array(vecs_en, dtype='float32'))
        faiss.write_index(idx_en, idx_en_path)
        with open(meta_en_path, 'w', encoding='utf-8') as mf:
            for blk in en_blocks: mf.write(json.dumps(blk, ensure_ascii=False)+"\n")
        st.sidebar.success("✅ 영어 인덱스 생성 완료")

    st.session_state.en_index = idx_en
    st.session_state.ko_blocks = ko_blocks
    st.session_state.en_blocks = en_blocks
    st.session_state.uploaded_files_key = key
    st.session_state.messages = []
    st.sidebar.info("4️⃣ 채팅 입력 대기 중…")

# ────────────────────────── 메인 채팅 UI ──────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m['role']): st.markdown(m['content'])

if ko_file and en_file and (query_ko := st.chat_input("질문을 한글로 입력하세요…")):
    st.session_state.messages.append({'role':'user','content':query_ko})
    with st.chat_message('user'): st.markdown(query_ko)

    query_en = translate_ko2en(query_ko)
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

    tag_prompt = (
        "Below is a list of possible Minecraft-related tags. "
        "Choose **1 to 4** tags most relevant to the question, and output them as a comma-separated list (no explanation):\n\n"
        f"Question (English): {query_en}\n\n"
        "Available tags:\n" + ", ".join(TAG_LIST) + "\n\n"
        "Answer:"                        # ★ 반드시 끝에 붙여주세요
    )


    # LLM에 태그 추출 요청
    tag_inputs = st.session_state.tok_chat(tag_prompt, return_tensors="pt").to(DEVICE_GPU)
    with torch.no_grad():
        tag_out_ids = st.session_state.mdl_chat.generate(
            **tag_inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
        )

    raw_tags = st.session_state.tok_chat.decode(
       tag_out_ids[0, tag_inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()
    # 쉼표로 분리하고, 리스트에 없는 건 제거, 최대 4개 취함
    st.write("🟢  선택된 태그:", raw_tags)
    selected_tags = [t.strip() for t in raw_tags.split(",") if t.strip() in TAG_LIST][:4]
    # 영어 검색용 질의에 태그 추가
    if selected_tags:
        query_en_with_tags = query_en + " | tags: " + ", ".join(selected_tags)
    else:
        query_en_with_tags = query_en




    with st.chat_message('assistant'):
        st.markdown("**1) 번역된 질문 (영어)**")
        st.write(query_en)

        st.markdown("**1.5) 자동 선택된 태그**")
        st.write(selected_tags or "없음")

        st.markdown("**2) 영어 RAG 후보 (상위 10)**")
        q_vec = st.session_state.embedder_cpu.embed_query(query_en_with_tags).reshape(1, -1)
        st.write("🔍 실제 검색용 쿼리:", query_en_with_tags)

        if 'en_index' in st.session_state and st.session_state.en_index is not None:
            st.write(f"Debug: FAISS index ntotal: {st.session_state.en_index.ntotal}")
        else:
            st.write("Debug: FAISS index (en_index) is not initialized.")

        if 'en_blocks' in st.session_state and st.session_state.en_blocks is not None:
            st.write(f"Debug: Length of en_blocks: {len(st.session_state.en_blocks)}")
        else:
            st.write("Debug: en_blocks is not initialized or is None.")

        D,I=st.session_state.en_index.search(np.array(q_vec,dtype='float32'),k=min(50,st.session_state.en_index.ntotal))
        TOP_CTX = 3                       # 컨텍스트로 쓸 개수
        ko_contexts = []
        for block_idx in I[0][:TOP_CTX]:
            if 0 <= block_idx < len(st.session_state.ko_blocks):
                ko_contexts.append(st.session_state.ko_blocks[block_idx])

        st.write(f"Debug: FAISS returned indices I[0]: {I[0][:10]}") 

        en_candidates_data = [] # 각 요소: {'text': 영어 청크, 'original_block_idx': en_blocks에서의 원본 인덱스}
        for score, original_block_idx_val in zip(D[0],I[0]):
            if 0 <= original_block_idx_val < len(st.session_state.en_blocks):
                txt = st.session_state.en_blocks[original_block_idx_val]
                en_candidates_data.append({'text': str(txt), 'original_block_idx': original_block_idx_val})
                snippet = str(txt).replace("\n"," ")[:100]+"…"
                st.write(f"- score={score:.4f}, chunk={snippet} (original_idx: {original_block_idx_val})")
            else:
                st.write(f"Debug: Invalid original_block_idx_val {original_block_idx_val} skipped.")

        if not en_candidates_data:
            st.warning("관련된 영어 후보 문장을 찾지 못했습니다.")
            st.session_state.messages.append({'role': 'assistant', 'content': "죄송합니다, 관련된 정보를 찾을 수 없습니다."})

        st.markdown("**4) 최종 LLM 프롬프트**")
        prompt = (
            "당신은 마인크래프트 아이템 전문 AI입니다.\n"
            "대답 내용은 아래에 있는 컨텍스트를 기반으로만 대답하고 컨텍스트에 없는 내용은 활용하지마세요.\n"
            "[CONTEXT]\n" + "\n\n".join(ko_contexts) + "\n\n"
            "[QUESTION]\n" + query_ko + "\n"
            "[ANSWER]"
        )
        st.code(prompt, language='text')

        # 답변 생성
        handler = StreamHandler(st.empty())
        answer = ""
        st.write("💬 답변 생성 중…")
        for tok in stream_generate(prompt): # stream_generate 함수가 정의되어 있다고 가정
            answer += tok
            handler.on_llm_new_token(tok)
        handler.on_llm_end(None)
        st.session_state.messages.append({'role':'assistant','content':answer})
