import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 사용할 엑사원 모델 이름 (사용자님의 환경에 맞게 확인해주세요)
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

def load_model_and_tokenizer(model_name):
    """모델과 토크나이저를 로드하는 함수"""
    print(f"Loading tokenizer for '{model_name}'...")
    # EXAONE 모델의 경우 trust_remote_code=True가 필요할 수 있습니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model '{model_name}'...")
    # 기본적인 모델 로딩 (GPU 사용 가능 시 GPU로, 아니면 CPU로)
    # 메모리 부족 시, BitsAndBytesConfig를 사용한 양자화 로딩 필요
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # 토크나이저의 pad_token_id가 설정되어 있지 않으면 eos_token_id로 설정
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 모델 설정에도 반영 (일부 모델에서 필요)
        if hasattr(model.config, 'pad_token_id'):
             model.config.pad_token_id = model.config.eos_token_id


    return model, tokenizer

def translate_korean_to_english(korean_text, model, tokenizer, device):
    """주어진 한국어 텍스트를 엑사원 모델을 사용하여 영어로 번역하는 함수"""

    prompt = f"""다음 한국어 문장을 영어로 번역해주세요.

한국어: "{korean_text}"
영어:"""

    print("\n번역 중...")
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False).to(device)
        input_ids_length = inputs.input_ids.shape[1]

        # 모델을 사용하여 번역 생성
        with torch.no_grad(): # 추론 시에는 그래디언트 계산 비활성화
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask, # 명시적으로 attention_mask 전달
                max_new_tokens=100,  # 번역될 문장의 최대 새 토큰 수 (필요에 따라 조절)
                num_beams=5,         # 빔 서치 사용 (더 나은 품질을 위해)
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id, # 패딩 토큰 ID 명시
                do_sample=False      # 일관된 출력을 위해 샘플링 비활성화
            )

        # 생성된 토큰에서 입력 프롬프트 부분을 제외하고 디코딩
        translated_tokens = outputs[0][input_ids_length:]
        translated_text = tokenizer.decode(translated_tokens, skip_special_tokens=True).strip()
        
        return translated_text
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return "번역 실패"

if __name__ == "__main__":
    # 사용 가능한 경우 GPU 사용, 그렇지 않으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = None
    tokenizer = None

    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
        model.to(device)
        model.eval()  # 모델을 평가 모드로 설정
    except Exception as e:
        print(f"모델 로딩 중 심각한 오류 발생: {e}")
        print("모델 이름이 정확한지, 모델에 접근 가능한지 확인해주세요.")
        print("메모리가 부족한 경우, RAG 코드에서 사용한 양자화(BitsAndBytesConfig)를 적용해야 할 수 있습니다.")
        exit()

    print("\nEXAONE 한국어 -> 영어 번역 테스트 CLI")
    print("번역할 한국어 문장을 입력해주세요. (종료하려면 'exit' 또는 'quit' 입력)")

    while True:
        korean_input = input("\n한국어 입력: ")
        if korean_input.lower() in ['exit', 'quit']:
            print("프로그램을 종료합니다.")
            break
        if not korean_input.strip():
            continue

        english_translation = translate_korean_to_english(korean_input, model, tokenizer, device)
        
        print(f"\n원본 (한국어): {korean_input}")
        print(f"번역 (영어): {english_translation}")
        print("------------------------------------")