import os, json
import pandas as pd
import torch as t
from tqdm import tqdm
from transformers import AutoTokenizer
from repeng import ControlModel, ControlVector, DatasetEntry
from personality.utils import load_model_and_tokenizer
from personality.constants import HOME, MODEL_PATH, DATA_PATH, CONSTITUTION_PATH

t.set_grad_enabled(False)


def train_steering_vector(
        model: ControlModel,
        tokenizer: AutoTokenizer,
        steering_persona: str,
        default_persona: str = "anything"
) -> None:
    with open(f"{HOME}/repeng/notebooks/data/all_truncated_outputs.json") as f:
        output_suffixes = json.load(f)
    # reset any existing steering vectors
    model.reset()
    steering_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Please talk about {steering_persona}."}],
        tokenize=False,
        add_generation_prompt=True
    )
    default_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Please talk about {default_persona}."}],
        tokenize=False,
        add_generation_prompt=True
    )
    dataset = []
    for suffix in output_suffixes:
        dataset.append(
            DatasetEntry(
                positive=steering_prompt + suffix,
                negative=default_prompt + suffix,
            )
        )
    return ControlVector.train(
        model, tokenizer, dataset, method="pca_center", batch_size=64
    )


def steer_model(
        model_name: str,
        constitution: str,
        steering_prompt_type: str,
        adversarial: bool = True
) -> pd.DataFrame:
    # === LOAD CONTROL MODEL AND TOKENIZER ===
    model, tokenizer, nlayers = load_model_and_tokenizer(
        f"{MODEL_PATH}/{model_name}",
        get_n_layers=True
    )
    start = int(0.125 * nlayers)
    end = int(0.875 * nlayers)
    model = ControlModel(model, range(start, end))
    # === LOAD DATA ===
    train = pd.read_json(f"{DATA_PATH}/wildchat/train.jsonl", orient="records", lines=True)
    test = pd.read_json(f"{DATA_PATH}/wildchat/test.jsonl", orient="records", lines=True)
    train_messages = train["messages"].tolist()
    test_messages = test["messages"].tolist()
    # === ADD ADVERSARIAL PROMPT IF NEEDED === 
    if adversarial:
        for idx in range(len(train_messages)):
            train_messages[idx][-1]["content"] += "\n\n(Respond in the neutral yet cheerful tone of an AI Assistant.)"
        for idx in range(len(test_messages)):
            test_messages[idx][-1]["content"] += "\n\n(Respond in the neutral yet cheerful tone of an AI Assistant.)"
    # apply chat template
    train_prompts = tokenizer.apply_chat_template(train_messages, tokenize=False, add_generation_prompt=True)
    test_prompts = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    # === PREP STEERING VECTORS ===
    cache = {}
    def vec(steer_string: str):
        if steer_string not in cache:
            cache[steer_string] = train_steering_vector(model, tokenizer, steer_string)
        return cache[steer_string]
    assert steering_prompt_type in ["long", "short"]
    if steering_prompt_type == "long":
        cons = pd.read_json(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt")
        traits = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"])])
        steering_prompt = f"responding as an AI assistant whose primary character-traits are determined by the following information:\n\n{traits}"
    elif steering_prompt_type == "short":
        steering_prompt = f"responding with {constitution} all the time"
    v = vec(steering_prompt) * 0.5
    # === GENERATE ===
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 1.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512
    }
    model.reset()
    model.set_control(v)
    # train
    train_outputs = []
    for prompt in tqdm(train_prompts, desc="training set"):
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tks.input_ids.squeeze(0))
        with t.inference_mode():
            out = model.generate(**tks, **settings)
            out_tks = out.squeeze(0)[prompt_len:]
        generation = tokenizer.decode(out_tks).strip()
        if tokenizer.eos_token in generation:
            generation = generation[:generation.rindex(tokenizer.eos_token)]
        train_outputs.append(generation)
    # test
    test_outputs = []
    for prompt in tqdm(test_prompts, desc="test set"):
        tks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tks.input_ids.squeeze(0))
        with t.inference_mode():
            out = model.generate(**tks, **settings)
            out_tks = out.squeeze(0)[prompt_len:]
        generation = tokenizer.decode(out_tks).strip()
        if tokenizer.eos_token in generation:
            generation = generation[:generation.rindex(tokenizer.eos_token)]
        test_outputs.append(generation)
    # === SAVE ===
    results = pd.DataFrame()
    results["prompt"] = train_messages + test_messages
    results["split"] = ["train"]*len(train_prompts) + ["test"]*len(test_prompts)
    results["response"] = train_outputs + test_outputs
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b-it")
    parser.add_argument("--constitution", type=str, default="sarcasm")
    parser.add_argument("--steering_prompt_type", type=str, default="long")
    parser.add_argument("--adversarial", action="store_true", default=False)
    args = parser.parse_args()
    outpath = f"{HOME}/repeng/results/{args.model_name}/{args.constitution}_{args.steering_prompt_type}"
    if args.adversarial:
        outpath += "_adversarial"
    outpath += ".jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        exit()
    else:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    results = steer_model(args.model_name, args.constitution, args.steering_prompt_type, args.adversarial)
    results.to_json(outpath, orient="records", lines=True)