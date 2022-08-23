import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TopPLogitsWarper, TopKLogitsWarper, LogitsProcessorList
import pandas as pd
from datasets import load_dataset

import pickle as pkl
import argparse
import os
import numpy as np

# set DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_src(path, sep):
    '''loads lines into memory from file'''
    with open(path, 'r') as f:
        for line in f:
            prompt, story = line.split(sep)
            yield prompt + sep


def load_ref(path, sep):
    '''loads lines into memory from file'''
    with open(path, 'r') as f:
        for line in f:
            prompt, story = line.split(sep)
            yield [story]


def load_gens(path, max_lines):
    data = pkl.load(open(path, 'rb'))
    df = pd.DataFrame({decoder: data[decoder]['text'][:max_lines]
                       for decoder in data if decoder != 'bayes_raw'})
    print([decoder for decoder in data if decoder != 'bayes_raw'])
    for row in df.iterrows():
        yield list(row[1])


def log_softmax(logits, decoder_input_ids, temp=1.0):

    normalized = torch.nn.functional.log_softmax(logits, dim=-1)

    # gather
    log_probs = normalized.gather(-1,
                                  decoder_input_ids.unsqueeze(-1)).squeeze(-1)

    return log_probs


def entropy(logits, temp=1.0):
    logp = torch.nn.functional.log_softmax(logits, dim=-1)

    p = torch.exp(logp)

    plogp = logp * p
    return -plogp.sum(-1)


def calc_logits(model, tokenizer, src_iterator, ref_iterator, top_p=None, top_k=None):
    log_probs = []
    tokens = []
    text = []
    ent_list = []
    p_nums = []
    epsilon_list = []
    for src, refs in zip(src_iterator, ref_iterator):
        with torch.no_grad():

            for ref in refs:

                combined = src + ref
                # cannot batch decode because padding  adds dummy probabilites that are hard to get rid off
                decoder_inputs = tokenizer(combined, truncation=True, padding='longest',
                                           max_length=1024, return_tensors='pt').to(DEVICE)

                src_encoding = tokenizer(src, truncation=True, padding='longest',
                                         max_length=1024, return_tensors='pt').to(DEVICE)

                labels = decoder_inputs.input_ids.clone()
                labels[:, :src_encoding.input_ids.shape[-1]] = -100

                forward = model(**decoder_inputs,
                                labels=labels, return_dict=True)

                shift_logits = forward.logits[..., :-1, :].contiguous()
                shift_labels = decoder_inputs.input_ids[..., 1:].contiguous()

                # warp logits
                processor = LogitsProcessorList()
                if top_p:
                    processor.append(TopPLogitsWarper(top_p=top_p))
                if top_k:
                    processor.append(TopKLogitsWarper(top_k=top_k))

                shift_logits = processor(
                    shift_labels.squeeze(0), shift_logits.squeeze(0)).unsqueeze(0)

                # calculate entropy
                normalized = torch.nn.functional.log_softmax(
                    shift_logits, dim=-1)
                p = torch.exp(normalized)
                ent = -(normalized * p).sum(-1, keepdim=True)

                #shift and sort
                diff = (-ent) - normalized
                shifted_scores = torch.abs(diff)
                sorted_shifted_logits, sorted_indices = torch.sort(
                    shifted_scores, descending=False)
                sorted_logits = shift_logits.gather(-1, sorted_indices)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

                # calculate the value of p
                p_num = cumulative_probs.scatter(-1, sorted_indices,
                                                 cumulative_probs).gather(-1, shift_labels.unsqueeze(-1))
                p_nums.append(
                    p_num[:, (src_encoding.input_ids.shape[-1] - 1):].squeeze().to('cpu'))
                # calculate epsilon
                eps = diff.gather(-1, shift_labels.unsqueeze(-1))[
                    :, (src_encoding.input_ids.shape[-1] - 1):]
                epsilon_list.append(eps.squeeze().to('cpu'))

                shift_ents = ent[:, (src_encoding.input_ids.shape[-1] - 1):]

                # save
                log_probs.append(log_softmax(
                    shift_logits, shift_labels)[:, (src_encoding.input_ids.shape[-1] - 1):].to('cpu'))
                tokens.append(decoder_inputs.input_ids.to('cpu'))
                text.append(ref)
                ent_list.append(shift_ents.to('cpu'))
    return {'log_probs': log_probs, 'tokens': tokens, 'text': text, 'entropy': ent_list, 'p': p_nums,
            'epsilon': epsilon_list}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help='output directory')
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('--num_gens', type=int,
                        default=1, help='number of generations per prompt')
    parser.add_argument('--refs', action='store_true',
                        help='forward pass references, if not set forward pass generations instead')
    parser.add_argument('--top_p', type=float,
                        help='if set to a value 0 < top_p < 1 warp the logits accordingly')
    parser.add_argument('--top_k', type=int,
                        help='perform top_k logits warping')
    cl_args = parser.parse_args()

    output_path = cl_args.output or '../data/forward/stories'
    num_gens = cl_args.num_gens

    top_p = cl_args.top_p
    top_k = cl_args.top_k

    # load data
    ref_path = cl_args.input or '../data/datasets/writingPrompts/test.comb.txt'
    gen_path = '../data/generations/stories_medium.generations.final.p'
    split = 'test'
    sep = '<|seperator|>'
    n_lines = 200

    # load human ids
    human_ids = np.sort(pd.read_csv(
        '../data/human_scores/stories_medium.final.csv').example_id.unique())
    lines = np.array(list(load_src(ref_path, sep)))[human_ids][:n_lines]
    src_iterator = list(lines)

    #ref_iterator = load_gens(gen_path, 400)
    ref_iterator = load_ref(
        ref_path, sep) if cl_args.refs else load_gens(gen_path, n_lines * num_gens)
    #model_path = '../data/models/gpt2_wp_large'
    model_path = '../../models/gpt2_wp_medium/final'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    # init pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    res = calc_logits(model, tokenizer, src_iterator,
                      ref_iterator, top_p, top_k)
    # save
    filename = 'stories_refs_probs' if cl_args.refs else 'stories_gens_probs'
    #filename = 'stories_gens_probs.p'
    if top_p:
        filename += '_' + str(top_p)
    if top_k:
        filename += '_' + str(top_k)
    filename += '.p'
    output_path = os.path.join(output_path, filename)
    pkl.dump(res, open(output_path, 'wb'))


if __name__ == '__main__':
    main()
