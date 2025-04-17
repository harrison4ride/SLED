# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py
# Ref: https://github.com/voidism/DoLa

from utils.utils_truthfulqa import *
import transformers
from tqdm import tqdm, trange
import argparse
import json
import numpy as np
import warnings
from sled_decoding import SLED_DecodedLLM_TruthfulQA as SLED_DecodedLLM


transformers.logging.set_verbosity(40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="Data/TruthfulQA")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--post_softmax", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy","SLED", "dola"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=50)


    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''

    list_data_dict = load_truthfulqa_dataset(args.data_path)

    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)

    if args.decoding_method == "VanillaGreedy":
        if args.early_exit_layers is not None:
            warnings.warn("The 'early_exit_layers' argument should be None when using Vanilla greedy decoding.")
        print("Vanilla greedy decoding from the final layer", flush=True)
        mature_layer = None
        candidate_premature_layers = None


    else:
        if args.early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(llm.num_layers + 1)]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
        print(
            f"MODE: {args.decoding_method} decoding with the final layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]

    answers = []
    result_dict = {'question': [], 'model_scores': [], 'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}

    with torch.no_grad():
        for sample in tqdm(list_data_dict):
            # reference answers
            ref_best = format_best(sample['answer_best'])
            ref_true = split_multi_answer(sample['answer_true'])
            ref_false = split_multi_answer(sample['answer_false'])

            scores_true = []
            scores_false = []

            generate_kwargs = dict(mode=args.decoding_method, mature_layer=mature_layer,
                                   candidate_premature_layers=candidate_premature_layers,
                                   relative_top=args.relative_top, relative_top_value=args.relative_top_value,
                                   post_softmax=args.post_softmax,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)


            for temp_ans in ref_true:
                prompt, answer = build_prompt_and_answer(sample['question'], temp_ans)
                log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)
                scores_true.append(log_probs)


            for temp_ans in ref_false:
                prompt, answer = build_prompt_and_answer(sample['question'], temp_ans)
                log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)
                scores_false.append(log_probs)



            scores = MC_calcs(scores_true, scores_false, ref_true, ref_best)
            # check nan in mc1/2/3
            if np.isnan(scores['MC1']) or np.isnan(scores['MC2']) or np.isnan(scores['MC3']):
                import ipdb;

                ipdb.set_trace()

            result_dict['model_scores'].append(scores)
            result_dict['question'].append(sample)
            # update total scores
            result_dict['total_mc1'] += scores['MC1']
            result_dict['total_mc2'] += scores['MC2']
            result_dict['total_mc3'] += scores['MC3']



    result_dict['total_mc1'] /= len(result_dict['question'])
    result_dict['total_mc2'] /= len(result_dict['question'])
    result_dict['total_mc3'] /= len(result_dict['question'])


    print(f'Final MC1/2/3: \n{result_dict["total_mc1"]}, {result_dict["total_mc2"]}, {result_dict["total_mc3"]}')

    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(args.output_path, 'w') as f:
        json.dump(result_dict, f)