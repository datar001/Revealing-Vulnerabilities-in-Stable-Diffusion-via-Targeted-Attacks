import random
import numpy as np
from PIL import Image
import copy
import json
from typing import Any, Mapping
import torch
from synonym import is_english, get_token_english_mask
import pdb

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def nn_project(curr_embeds, embedding_layer, forbidden_mask=None, forbidden_set=None, english_mask=None):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape

        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = curr_embeds / curr_embeds.norm(dim=1, keepdim=True) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)

        sims = torch.mm(curr_embeds, embedding_matrix.transpose(0, 1))

        if forbidden_mask is not None:
            sims[:, forbidden_mask] = -1e+8
            forbidden_num = len(forbidden_set)
        else:
            forbidden_num = 0

        if english_mask is not None:
            sims[:, english_mask] = -1e+8

        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(sims, max(1, forbidden_num+1), dim=1, largest=True, sorted=False)  #
        queries_result_list = []
        for idx in range(seq_len):
            cur_query_idx_topk = sorted([[top_k_id.item(), top_k_value.item()] for top_k_id, top_k_value in
                                         zip(cos_scores_top_k_idx[idx], cos_scores_top_k_values[idx])],
                                        key=lambda x:x[1], reverse=True)
            for token_id, sim_value in cur_query_idx_topk:
                if token_id not in forbidden_set:
                    queries_result_list.append(token_id)
                    break
        nn_indices = torch.tensor(queries_result_list,
                                  device=curr_embeds.device).reshape((bsz, seq_len)).long()

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices

def forbidden_mask(forbidden_words, tokenizer, device):

    mask = torch.zeros(len(tokenizer.encoder)).bool().to(device)
    squeeze_forbidden = set()
    if forbidden_words is not None:
        forbidden_token = [tokenizer._convert_token_to_id(word) for word in forbidden_words]
        forbidden_token.extend([tokenizer._convert_token_to_id(word + "</w>") for word in forbidden_words])
        for token in forbidden_token:
            squeeze_forbidden.add(token)
            mask[token] = 1
    return mask, squeeze_forbidden


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer):
    input_ids = input_ids.detach().cpu().numpy()
    texts = []
    token_text = []
    for ids in input_ids:
        tokens = [tokenizer._convert_id_to_token(int(id_)) for id_ in ids]
        texts.append(tokenizer.convert_tokens_to_string(tokens))
        token_text.append([token.replace("</w>", " ") for token in tokens])

    return texts, token_text


def get_target_feature(model, preprocess, tokenizer, device, target_images=None, target_prompts=None):
    if target_images is not None:
        with torch.no_grad():
            images = preprocess(images=target_images, return_tensors="pt").pixel_values
            image_features = model.get_image_features(pixel_values=images.to(device))
            # normalized features
            all_target_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    else:
        text_input = tokenizer(
            target_prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            all_target_features = model.get_text_features(text_input.input_ids.to(device))
            all_target_features = all_target_features / all_target_features.norm(p=2, dim=-1, keepdim=True)
    return all_target_features

def get_id(prompt, suffix_num, tokenizer, device, token_embedding):

    dummy_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer._tokenize(prompt)]
    prompt_len = len(dummy_ids)
    assert prompt_len + suffix_num < 76, "opti_num + len(prompt) must < 77"
    padded_template_text = '{}'.format(" ".join(["<|startoftext|>"] * (suffix_num)))
    # <|startoftext|> for transformer clip Autotokenizer
    # <start_of_text> for openclip

    # dummy_ids.extend(tokenizer.encode(padded_template_text))
    dummy_ids.extend([tokenizer.encoder[token] for token in tokenizer._tokenize(padded_template_text)])
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    # dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids]).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()  # get embedding of initial template, no grad
    dummy_embeds.requires_grad = False

    opti_num = (dummy_ids == -1).sum()
    prompt_ids = torch.randint(len(tokenizer.encoder), (1, opti_num)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    return prompt_embeds, dummy_embeds, dummy_ids, prompt_len + suffix_num


def optimize_prompt_loop(model, tokenizer, all_target_features,
                         args, device, ori_prompt=None, forbidden_words=None,
                         suffix_num=10, object_mask=None, english_mask=None,
                         ori_feature=None, writer=None):
    assert ori_prompt is not None
    opt_iters = args.iter
    lr = args.lr
    weight_decay = args.weight_decay
    print_step = args.print_step
    batch_size = args.batch_size
    top_k = 50
    prompt_topk = [[-1, 0, "", ""] for _ in range(top_k)]
    save_prompts = []
    top_k_sim_min = 0.


    token_embedding = model.text_model.embeddings.token_embedding

    # forbidden
    forbidden_mask_, squeeze_forbidden = forbidden_mask(forbidden_words, tokenizer, device)

    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids, max_char_num = get_id(ori_prompt, suffix_num, tokenizer, device, token_embedding)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_sim = -1000 * args.loss_weight
    best_text = ""

    for step in range(opt_iters):
        # randomly sample sample images and get features
        if batch_size is None:
            target_features = all_target_features
        else:
            curr_indx = torch.randperm(len(all_target_features))
            target_features = all_target_features[curr_indx][0:batch_size]
            
        
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding,
                                                  forbidden_mask=forbidden_mask_, english_mask=english_mask,
                                                  forbidden_set=squeeze_forbidden)

        # get cosine similarity score with all target features
        with torch.no_grad():
            # padded_embeds = copy.deepcopy(dummy_embeds)
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _, mse_loss = model.forward_text_embedding(padded_embeds,
                                                                         dummy_ids,
                                                                         target_features,
                                                                         object_mask=object_mask,
                                                                         ori_feature=ori_feature)
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()  # max
            best_indx = scores_per_prompt.argmax().item()
        
        # tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True
        
        # padding
        # padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)
        
        logits_per_image, _, mse_loss = model.forward_text_embedding(padded_embeds,
                                                                     dummy_ids,
                                                                     target_features,
                                                                     object_mask=object_mask,
                                                                     ori_feature=ori_feature)
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        if object_mask is not None:
            loss = loss + mse_loss  * args.loss_weight
        if writer is not None:
            writer.add_scalar('loss', loss.item(), step)
        
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
        
        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        target_id = dummy_ids.detach().clone()
        target_id[dummy_ids == -1] = nn_indices.reshape(1, -1)
        target_id = target_id[:, 1:max_char_num + 1]
        decoded_texts, decoded_tokens = decode_ids(target_id, tokenizer)
        decoded_text, decoded_token = decoded_texts[best_indx], decoded_tokens[best_indx]

        # save top k prompt
        if cosim_scores >= top_k_sim_min and decoded_text not in save_prompts:
            prompt_topk[0] = [cosim_scores, mse_loss.item(), decoded_text, decoded_token]
            prompt_topk = sorted(prompt_topk, key=lambda x:x[0])
            top_k_sim_min = prompt_topk[0][0]
            save_prompts.append(decoded_text)

        if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
            per_step_message = f"step: {step}, lr: {curr_lr}"
            per_step_message = f"\n{per_step_message}, " \
                               f"mse: {mse_loss.item():.3f}, " \
                               f"cosim: {cosim_scores:.3f}," \
                               f" text: {decoded_text}, " \
                                f"token: {decoded_token}"
            print(per_step_message)

        if best_sim * args.loss_weight < cosim_scores * args.loss_weight:
            best_sim = cosim_scores
            best_text = decoded_text

    return best_text, best_sim, prompt_topk


def optimize_prompt(model, preprocess, tokenizer, args, device, target_images=None, target_prompts=None, ori_prompt=None,
                    forbidden_words=None, suffix_num=10, object_mask=None, only_english_words=False, writer=None):

    # get target features
    all_target_features = get_target_feature(model, preprocess, tokenizer, device, target_images=target_images,
                                             target_prompts=target_prompts)
    # get original prompt feature
    with torch.no_grad():
        text_input = tokenizer(
            ori_prompt, padding=True, return_tensors="pt")
        ori_feature = model.get_text_features(text_input.input_ids.to(device))
        ori_feature = ori_feature / ori_feature.norm(p=2, dim=-1, keepdim=True)

    # only choose english
    if only_english_words:
        english_mask = get_token_english_mask(tokenizer.encoder, device)
    else:
        english_mask = None

    # optimize prompt
    learned_prompt = optimize_prompt_loop(model, tokenizer,
                                          all_target_features,
                                          args, device, ori_prompt=ori_prompt,
                                          forbidden_words=forbidden_words,
                                          suffix_num=suffix_num, object_mask=object_mask,
                                          english_mask=english_mask,
                                          ori_feature=ori_feature,
                                          writer=writer)

    return learned_prompt