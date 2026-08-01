import argparse
import json
import os
import glob
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "./results_rag"
SAMPLES_PATH = "./samples.json"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_NEW_TOKENS = 32


def clean_text(text):
    return " ".join(str(text).strip().split())


def parse_bool(text):
    text = text.strip().lower()

    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False
    if "true" in text and "false" not in text:
        return True
    if "false" in text and "true" not in text:
        return False

    return False


def build_prompt(question, answer, candidate):
    return (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Candidate: {candidate}\n"
        "Is candidate correct?"
    )


@torch.inference_mode()
def judge(tokenizer, model, question, answer, candidate):
    messages = [
        {
            "role": "system",
            "content": "You are a strict evaluator. Output only true or false.",
        },
        {
            "role": "user",
            "content": build_prompt(question, answer, candidate),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return parse_bool(output_text)


def load_llama():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def process_file(result_file, samples, tokenizer, model, model_key):
    dataset_key = "olmo" if model_key == "olmo32" else model_key

    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for qid in tqdm(results, desc=os.path.basename(result_file)):
        if qid not in samples[dataset_key]:
            continue

        ground_truth = clean_text(samples[dataset_key][qid]["obj"])

        for condition in results[qid]:
            question = clean_text(results[qid][condition]["question"])
            predicted_answer = clean_text(results[qid][condition]["answer"])

            results[qid][condition]["is_correct"] = judge(
                tokenizer=tokenizer,
                model=model,
                question=question,
                answer=ground_truth,
                candidate=predicted_answer,
            )

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[✓] Written back: {result_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["amber", "redpajama", "olmo", "olmo32"],
    )
    args = parser.parse_args()

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    pattern = os.path.join(RESULTS_DIR, f"{args.model}.json")
    result_files = sorted(glob.glob(pattern))

    print(f"[✓] Found {len(result_files)} files")
    for path in result_files:
        print(f"  - {path}")

    tokenizer, judge_model = load_llama()

    for result_file in result_files:
        process_file(
            result_file=result_file,
            samples=samples,
            tokenizer=tokenizer,
            model=judge_model,
            model_key=args.model,
        )


if __name__ == "__main__":
    main()