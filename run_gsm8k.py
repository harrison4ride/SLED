# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py
# Ref: https://github.com/voidism/DoLa
import transformers
from tqdm import tqdm, trange
import argparse
from utils.utils_gsm8k import *
from sled_decoding import SLED_DecodedLLM_GSM8K as SLED_DecodedLLM
import json
import warnings
transformers.logging.set_verbosity(40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str, default="./gsm8k")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy","SLED", "dola"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)


    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    set_seed(args.seed)
    list_data_dict = load_gsm8k_jsonl(args.data_path)


    
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:", "\end{code}"]
    llm.set_stop_words(stop_word_list)

    if args.decoding_method == "VanillaGreedy":
        if args.early_exit_layers is not None:
            warnings.warn("The 'early_exit_layers' argument should be None when using Vanilla greedy decoding.")
        print("Vanilla greedy decoding from the final layer", flush=True)
        mature_layer = None
        candidate_premature_layers = None

    else:
        if args.early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(llm.num_layers+1)]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]

        print(f"MODE: {args.decoding_method} decoding with the final layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]



    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    for sample in tqdm(list_data_dict[:]):
        input_text = build_prompt(sample['instruction'], args.do_shuffle)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p,
                               top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                               mode=args.decoding_method, mature_layer=mature_layer,
                               candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top,
                               relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)


    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(args.output_path , 'w') as f:
        json.dump(result_dict, f)

    print(f'Num of total question: {len(answers)}, '
         f'correct num: {sum(answers)}, '
         f'correct rate: {float(sum(answers)) / len(answers)}.')
