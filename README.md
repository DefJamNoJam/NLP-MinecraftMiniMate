# 🪄 Minecraft RAG Desktop Chatbot

> **“마인크래프트 아이템 – 궁금하면 켜고 바로 물어보세요!”**  
> • 한글 질문 → 영어 번역 → FAISS 검색 → EXAONE-3.5 LLM 답변  
> • **단일 실행 파일**(`gui_rag.exe`) 로 동작하는 오프라인 RAG 챗봇  

- 🔗 Embedding 모델: [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)  
- 🔗 LLM 모델: [`LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)

## 🛠️ 소스에서 직접 실행

```bash
# 1. 가상환경 설정 (최초 1회, 가상환경 이름은 mmm 이 아니어도 상관없음.)
conda create -n mmm_env python=3.10
conda activate mmm_env
pip install -r requirements.txt

# 2. 서버 실행 (백엔드 시작)
startserver.bat
# 서버 로그에 아래 문구가 출력되면 준비 완료:
# ➤ "Startup complete. Uvicorn running on http://0.0.0.0:8504"

# 3. Electron 앱 실행 (프론트엔드 UI)
start.bat
```

- `startserver.bat`은 `src/rag_server.py`를 실행하여 모델 로딩 및 RAG API 서버를 실행합니다.
- 서버가 완전히 시작되고 `"Startup complete"` 로그가 출력되면, `start.bat`을 실행해 데스크탑 앱 UI를 사용할 수 있습니다.
- 모든 실행은 **로컬에서 GPU 환경 기반**으로 작동하며, 인터넷 없이도 동작합니다.


## 🔎 파이프라인

| 단계 | Stage         | 설명 |
|------|---------------|------|
| 1    | Input         | GUI(.exe)에서 한글 질문 입력 |
| 2    | Translation   | `deep_translator.GoogleTranslator` → 영어 변환 |
| 3    | Tag 추출      | LLM 프롬프트로 연관 태그 1~4개 예측 |
| 4    | Embedding     | `all-mpnet-base-v2` (CPU)로 “질문 + 태그” 벡터화 |
| 5    | Retrieval     | FAISS Inner-product 검색 (k=20) |
| 6    | Mapping       | top-3 index → 대응 한글 블록 추출 |
| 7    | Prompt        | `[CONTEXT] ko₁ ko₂ ko₃ + [QUESTION] ko` 구성 |
| 8    | Generation    | EXAONE-3.5 (4-bit, GPU 우선) – max 256 tokens |
| 9    | Display/Log   | PySimpleGUI 출력 + `./logs/` 기록 |

---

## 📂 Repository Structure

```
gui_rag.py        # main script  
gui_rag.spec      # PyInstaller spec  
set_env.py        # KMP_DUPLICATE_LIB_OK runtime-hook  
mc_ko.txt         # Korean knowledge base  
mc_en.txt         # English knowledge base  
requirements.txt  # pip dependencies  
logs/             # execution logs  
```

---

## ✨ Roadmap / TODO

- Google API → 로컬 Marian MT / MiniLM NMT 교체  
- `all-mpnet-base-v2` 를 Minecraft wiki로 파인튜닝 → recall ↑  
- 공식 Wiki RSS 스크래핑으로 KB 자동 갱신  
- GUI에 GPU / CPU 토글 추가  
- Linux AppImage / macOS `.app` 빌드 지원  

---

## 📜 License

- 코드: MIT  
- 텍스트(`mc_*.txt`): Minecraft Wiki (CC BY-SA 4.0) 인용

```

## 📚 참고 자료

* [프로젝트 발표 자료 (PDF)](./docs/MinecraftMiniMate.pdf)

