# Ref: https://github.com/voidism/DoLa
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList#, LLamaQaStoppingCriteria
import numpy as np


class SLED_DecodedLLM_TruthfulQA:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

        self.num_layers = self.model.config.num_hidden_layers if hasattr(self.model.config,
                                                                         "num_hidden_layers") else self.model.config.n_layer

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1,
                                min_tokens_to_keep: int = 1):

        scores_normalized = scores.log_softmax(dim=-1)

        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        probs_max = torch.max(scores_normalized, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)

        probs_thresh = torch.min(min_thresh, probs_thresh)

        probs_thresh = probs_thresh.unsqueeze(-1)

        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False,
                 mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy',
                 verbose=True,
                 remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=False,
                 evolution_rate=2, evolution_scale=10, evolution_lower_bound=-100, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            if mode == 'VanillaGreedy':
                outputs = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    outputs = outputs.log_softmax(-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()


            elif mode == 'dola':
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack(
                        [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                     dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                                         dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :,
                               :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                             dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers,
                                                                 dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(
                        -1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
                        -1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()


            elif mode == 'SLED':
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )
                new_output_logits = dict_outputs[mature_layer].clone()

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    stacked_premature_layers = torch.stack(
                        [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                     dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                                         dim=-1)
                    topk_prob, topk_indices = torch.topk(softmax_mature_layer, evolution_scale)
                    topk_indices = topk_indices[0]

                    divergence = stacked_premature_layers - dict_outputs[mature_layer][:, seq_i, :]
                    candidate_gradients_expanded = softmax_premature_layers.expand(-1, len(topk_indices), -1)
                    candidate_mask = torch.zeros_like(candidate_gradients_expanded)
                    topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2)
                    candidate_mask.scatter_(2, topk_indices_expanded.expand(softmax_premature_layers.size(0), -1, -1),
                                            1)
                    candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask
                    candidate_gradients_expanded = candidate_gradients_expanded.to(torch.float32)
                    layer_divergence_expanded = divergence.to(torch.float32)

                    layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded,
                                                            dim=2)
                    layer_topk_values, layer_topk_indices = torch.topk(layer_dot_results, evolution_scale)
                    layer_topk_topk_indices = topk_indices[layer_topk_indices]

                    layer_topk_values = (layer_topk_values * (layer_topk_values > 0)) ** 2
                    layer_topk_values_sum_layers = torch.sum(layer_topk_values, dim=1).clone()
                    non_zero_indices = layer_topk_values_sum_layers != 0
                    layer_topk_values[non_zero_indices] /= layer_topk_values_sum_layers[non_zero_indices].unsqueeze(1)
                    if layer_topk_values_sum_layers.sum() != 0:
                        layer_topk_values_sum_layers = layer_topk_values_sum_layers / layer_topk_values_sum_layers.sum()
                    proxy_gradients_tensor_delta = torch.zeros_like(softmax_mature_layer,
                                                                    device=layer_divergence_expanded.device).to(
                        layer_divergence_expanded.dtype).repeat(layer_topk_values.size(0), 1)
                    proxy_gradients_tensor_delta.scatter_(1, layer_topk_topk_indices, -layer_topk_values)
                    proxy_gradients_tensor_delta = torch.sum(
                        proxy_gradients_tensor_delta * layer_topk_values_sum_layers.unsqueeze(1), dim=0)
                    proxy_gradients_tensor_delta = proxy_gradients_tensor_delta.to(softmax_mature_layer.dtype)
                    hidden_states_seq_i = new_output_logits[:, seq_i, :].clone()

                    op_T = 10
                    evolution_rate_values = [evolution_rate * (1 - i / op_T) for i in range(op_T)]

                    for op_t in range(op_T):
                        lr_t = evolution_rate_values[op_t]
                        softmax_hidden_states_seq_i = F.softmax(hidden_states_seq_i, dim=-1)
                        proxy_gradients_tensor = softmax_hidden_states_seq_i + proxy_gradients_tensor_delta
                        hidden_states_seq_i.sub_(lr_t * proxy_gradients_tensor)

                    hidden_states_seq_i_new = torch.full_like(hidden_states_seq_i[0], fill_value=evolution_lower_bound,
                                                              device=hidden_states_seq_i.device,
                                                              dtype=hidden_states_seq_i.dtype)
                    hidden_states_seq_i_new[topk_indices] = hidden_states_seq_i[0, topk_indices]
                    new_output_logits[:, seq_i, :] = hidden_states_seq_i_new.unsqueeze(dim=0)

                if post_softmax:
                    log_new_output_logits = F.log_softmax(new_output_logits, dim=-1)
                else:
                    log_new_output_logits = new_output_logits

                log_new_output_logits = log_new_output_logits[0, prefix_ids.shape[-1] - 1: -1, :]
                log_probs = log_new_output_logits[range(log_new_output_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'dola' else None)


class SLED_DecodedLLM_Factor:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)
        self.num_layers = self.model.config.num_hidden_layers if hasattr(self.model.config,
                                                                         "num_hidden_layers") else self.model.config.n_layer

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1,
                                min_tokens_to_keep: int = 1):

        scores_normalized = scores.log_softmax(dim=-1)

        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        probs_max = torch.max(scores_normalized, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)

        probs_thresh = torch.min(min_thresh, probs_thresh)

        probs_thresh = probs_thresh.unsqueeze(-1)

        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False,
                 mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy',
                 verbose=True,
                 remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                 evolution_rate=2, evolution_scale=10, evolution_lower_bound=-2500, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            if mode == 'VanillaGreedy':
                outputs = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    outputs = outputs.log_softmax(-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()


            elif mode == 'dola':
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack(
                        [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                     dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                                         dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :,
                               :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                             dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers,
                                                                 dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(
                        -1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
                        -1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()


            elif mode == 'SLED':
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )
                new_output_logits = dict_outputs[mature_layer].clone()

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    stacked_premature_layers = torch.stack(
                        [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                     dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                                         dim=-1)
                    topk_prob, topk_indices = torch.topk(softmax_mature_layer, evolution_scale)
                    topk_indices = topk_indices[0]

                    divergence = stacked_premature_layers - dict_outputs[mature_layer][:, seq_i, :]
                    candidate_gradients_expanded = softmax_premature_layers.expand(-1, len(topk_indices), -1)
                    candidate_mask = torch.zeros_like(candidate_gradients_expanded)
                    topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2)
                    candidate_mask.scatter_(2, topk_indices_expanded.expand(softmax_premature_layers.size(0), -1, -1),
                                            1)
                    candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask
                    candidate_gradients_expanded = candidate_gradients_expanded.to(torch.float32)
                    layer_divergence_expanded = divergence.to(torch.float32)

                    layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded,
                                                            dim=2)
                    layer_topk_values, layer_topk_indices = torch.topk(layer_dot_results, evolution_scale)
                    layer_topk_topk_indices = topk_indices[layer_topk_indices]

                    layer_topk_values = (layer_topk_values * (layer_topk_values > 0)) ** 2
                    layer_topk_values_sum_layers = torch.sum(layer_topk_values, dim=1).clone()
                    non_zero_indices = layer_topk_values_sum_layers != 0
                    layer_topk_values[non_zero_indices] /= layer_topk_values_sum_layers[non_zero_indices].unsqueeze(1)
                    if layer_topk_values_sum_layers.sum() != 0:
                        layer_topk_values_sum_layers = layer_topk_values_sum_layers / layer_topk_values_sum_layers.sum()
                    proxy_gradients_tensor_delta = torch.zeros_like(softmax_mature_layer,
                                                                    device=layer_divergence_expanded.device).to(
                        layer_divergence_expanded.dtype).repeat(layer_topk_values.size(0), 1)
                    proxy_gradients_tensor_delta.scatter_(1, layer_topk_topk_indices, -layer_topk_values)
                    proxy_gradients_tensor_delta = torch.sum(
                        proxy_gradients_tensor_delta * layer_topk_values_sum_layers.unsqueeze(1), dim=0)
                    proxy_gradients_tensor_delta = proxy_gradients_tensor_delta.to(softmax_mature_layer.dtype)
                    hidden_states_seq_i = new_output_logits[:, seq_i, :].clone()

                    op_T = 1
                    evolution_rate_values = [evolution_rate * (1 - i / op_T) for i in range(op_T)]

                    for op_t in range(op_T):
                        lr_t = evolution_rate_values[op_t]
                        softmax_hidden_states_seq_i = F.softmax(hidden_states_seq_i, dim=-1)
                        proxy_gradients_tensor = softmax_hidden_states_seq_i + proxy_gradients_tensor_delta
                        hidden_states_seq_i.sub_(lr_t * proxy_gradients_tensor)

                    hidden_states_seq_i_new = torch.full_like(hidden_states_seq_i[0], fill_value=evolution_lower_bound,
                                                              device=hidden_states_seq_i.device,
                                                              dtype=hidden_states_seq_i.dtype)
                    hidden_states_seq_i_new[topk_indices] = hidden_states_seq_i[0, topk_indices]
                    new_output_logits[:, seq_i, :] = hidden_states_seq_i_new.unsqueeze(dim=0)

                if post_softmax:
                    log_new_output_logits = F.log_softmax(new_output_logits, dim=-1)
                else:
                    log_new_output_logits = new_output_logits

                log_new_output_logits = log_new_output_logits[0, prefix_ids.shape[-1] - 1: -1, :]
                log_probs = log_new_output_logits[range(log_new_output_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'dola' else None)


class SLED_DecodedLLM_GSM8K:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)
        self.num_layers = self.model.config.num_hidden_layers if hasattr(self.model.config,
                                                                         "num_hidden_layers") else self.model.config.n_layer

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        # else:
        #     raise ValueError(f"Invalid device: {self.device}")

        if model_name =='deepseek-ai/DeepSeek-V2-Lite-Chat' or model_name =='mistralai/Codestral-22B-v0.1' or model_name =='huggyllama/llama-13b' or model_name =='google/gemma-2-27b-it':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,trust_remote_code=True,torch_dtype=torch.float16)
        elif model_name !='huggyllama/llama-7b':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
            model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

        if 'cuda' in self.device and int(self.num_gpus) == 1:
            # model.cuda()
            model.to(torch.device(self.device))
            print('loading model to',self.device)
        # if self.device == "cuda" and self.num_gpus == 1:
        #     model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        # self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None,
                 premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy', verbose=True,
                 remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                 evolution_rate=2, evolution_scale=10, evolution_lower_bound=-1000, **kwargs):

        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'VanillaGreedy':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, **kwargs)

            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, premature_layer=None,
                                              candidate_premature_layers=candidate_premature_layers, **kwargs, )
                premature_layer_dist = outputs.premature_layer_dist


            elif mode == 'SLED':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, 
                                              candidate_premature_layers=candidate_premature_layers,
                                              relative_top_value=relative_top_value, sled_decoding=True,
                                              evolution_rate=evolution_rate, evolution_scale=evolution_scale,
                                              evolution_lower_bound=evolution_lower_bound, **kwargs, )
                premature_layer_dist = None

            sequences, scores = outputs.sequences, outputs.scores
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1,
                                min_tokens_to_keep: int = 1):

        scores_normalized = scores.log_softmax(dim=-1)

        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        probs_max = torch.max(scores_normalized, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)

        probs_thresh = torch.min(min_thresh, probs_thresh)

        probs_thresh = probs_thresh.unsqueeze(-1)

        return scores_normalized < probs_thresh


class SLED_DecodedLLM_StrQA:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)
        self.num_layers = self.model.config.num_hidden_layers if hasattr(self.model.config,
                                                                         "num_hidden_layers") else self.model.config.n_layer

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        # else:
        #     raise ValueError(f"Invalid device: {self.device}")

        if model_name =='deepseek-ai/DeepSeek-V2-Lite-Chat' or model_name =='mistralai/Codestral-22B-v0.1' or model_name =='huggyllama/llama-13b' or model_name =='google/gemma-2-27b-it':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,trust_remote_code=True,torch_dtype=torch.float16)
        elif model_name !='huggyllama/llama-7b':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
            model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

        if 'cuda' in self.device and int(self.num_gpus) == 1:
            # model.cuda()
            model.to(torch.device(self.device))
            print('loading model to',self.device)
        # if self.device == "cuda" and self.num_gpus == 1:
        #     model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            if self.model_name.startswith("meta-llama/Meta-Llama-3"):
                if stop_word == "Q:":
                    stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
                else:
                    stop_word_ids = self.tokenizer.encode(stop_word)[1:]

            else:
                stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        # self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None,
                 premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy', verbose=True,
                 remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                 evolution_rate=2, evolution_scale=10, evolution_lower_bound=-1000, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'VanillaGreedy':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, **kwargs)

            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, premature_layer=None,
                                              candidate_premature_layers=candidate_premature_layers, **kwargs, )
                premature_layer_dist = outputs.premature_layer_dist


            elif mode == 'SLED':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, 
                                              candidate_premature_layers=candidate_premature_layers,
                                              relative_top_value=relative_top_value, sled_decoding=True,
                                              evolution_rate=evolution_rate, evolution_scale=evolution_scale,
                                              evolution_lower_bound=evolution_lower_bound, **kwargs, )
                premature_layer_dist = None #outputs.premature_layer_dist

            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1,
                                min_tokens_to_keep: int = 1):

        scores_normalized = scores.log_softmax(dim=-1)

        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        probs_max = torch.max(scores_normalized, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)

        probs_thresh = torch.min(min_thresh, probs_thresh)

        probs_thresh = probs_thresh.unsqueeze(-1)

        return scores_normalized < probs_thresh
