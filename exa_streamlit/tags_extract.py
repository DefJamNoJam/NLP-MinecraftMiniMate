#!/usr/bin/env python3
# filename: extract_tags_with_exaone.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ────────────────────────── 설정 ──────────────────────────
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# quantization 설정은 상황에 따라 조정하세요
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

# ────────────────────────── 모델 로드 ──────────────────────────
print("모델 로드 중…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map={"": 0} if torch.cuda.is_available() else {"": -1},
)
print("모델 로드 완료.")

# ────────────────────────── 태그 목록 ──────────────────────────
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

def extract_tags(question_en: str) -> list[str]:
    tag_prompt = (
        "Below is a list of possible Minecraft-related tags. "
        "Choose **1 to 4** tags most relevant to the question, and output them as a comma-separated list (no explanation):\n\n"
        f"Question (English): {question_en}\n\n"
        "Available tags:\n" + ", ".join(TAG_LIST) + "\n\n"
        "Answer:"
    )

    inputs = tok(tag_prompt, return_tensors="pt").to(mdl.device)
    out_ids = mdl.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
    )
    raw = tok.decode(
        out_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    # comma split, trim, filter to known tags, take up to 4
    tags = [t.strip() for t in raw.split(",")]
    return [t for t in tags if t in TAG_LIST][:4]

def main():
    if len(sys.argv) > 1:
        question_en = " ".join(sys.argv[1:])
    else:
        question_en = input("질문을 영어로 입력하세요: ").strip()
    tags = extract_tags(question_en)
    if tags:
        print("추출된 태그:", ", ".join(tags))
    else:
        print("추출된 태그가 없습니다.")

if __name__ == "__main__":
    main()
