import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)

# ======================
# STATIC CONFIG
# ======================

SAMPLES_PATH = "./samples.json"
OUTPUT_DIR = "./results_rag"

MAX_NEW_TOKENS = 128

DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0

# ======================


MODEL_MAP = {
    "amber": "LLM360/AmberChat",
    "redpajama": "togethercomputer/RedPajama-INCITE-7B-Instruct",
    "olmo": "allenai/Olmo-3-7B-Instruct",
    "olmo32": "allenai/Olmo-3.1-32B-Instruct",
}

AMBER_TEMPLATE = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
### Human: Got any creative ideas for a 10 year old’s birthday?
### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments.
3. Outdoor Movie Night: Set up a backyard movie night with a projector.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity.
5. Sports Olympics: Host a mini Olympics event.
6. Cooking Party: Have a cooking-themed party.
7. Superhero Training Camp: Create a superhero-themed party.
8. Outdoor Adventure: Plan an outdoor adventure party.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!
### Human: {prompt}
### Assistant:"""


def clean_answer(text: str) -> str:
    text = text.strip()

    stop_markers = [
        "### Human:",
        "### Assistant:",
        "\nQuestion:",
        "\nAnswer:",
        "</s>",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    text = text.split("\n")[0].strip()
    text = text.strip(" .,:;\"'")

    return text


def load_model(model_key: str):
    model_name = MODEL_MAP[model_key]

    if model_key == "amber":
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model.eval()

    return tokenizer, model


def format_prompt(
        model_key: str,
        tokenizer,
        question: str,
        context: str = None,
):
    if context is not None:
        prompt_text = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    else:
        prompt_text = (
            f"Question: {question}\n"
            f"Answer:"
        )

    if model_key == "amber":
        return AMBER_TEMPLATE.format(prompt=prompt_text)

    if model_key == "redpajama":
        return prompt_text

    if model_key in {"olmo", "olmo32"}:
        messages = [
            {
                "role": "user",
                "content": (
                    f'Do not say "As of my last update" and something like it. '
                    f"Just Answer.\n\n"
                    f"{prompt_text}\n"
                ),
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    raise ValueError(f"Unknown model: {model_key}")


@torch.no_grad()
def generate(
        tokenizer,
        model,
        prompt,
):
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        generation_kwargs["temperature"] = TEMPERATURE
        generation_kwargs["top_p"] = TOP_P

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]

    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return clean_answer(decoded)


def load_existing_results(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_result(
        results,
        qid,
        condition,
        question,
        prompt,
        answer,
):
    results[qid][condition] = {
        "question": question,
        "prompt": prompt,
        "answer": answer,
    }


def run(args):
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    model_key = args.model

    dataset_key = "olmo" if model_key == "olmo32" else model_key

    data = samples[dataset_key]

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_key}.json"

    results = load_existing_results(output_path)

    tokenizer, model = load_model(model_key)

    for qid, item in tqdm(data.items()):
        if qid not in results:
            results[qid] = {}

        question = item["questions"][args.question_type]

        conditions = {
            "no_context": None,
            "lexical_so": item["passages"]["lexical_so"],
            "lexical_sro": item["passages"]["lexical_sro"],
            "support": item["passages"]["support"],
            "irrelevant": item["passages"]["irrelevant"],
        }

        for condition, context in conditions.items():
            prompt = format_prompt(
                model_key=model_key,
                tokenizer=tokenizer,
                question=question,
                context=context,
            )

            answer = generate(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
            )

            save_result(
                results=results,
                qid=qid,
                condition=condition,
                question=question,
                prompt=prompt,
                answer=answer,
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Saved results to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
    )

    parser.add_argument(
        "--question_type",
        type=str,
        required=True,
        choices=[
            "template_based",
            "simple",
            "complex",
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
