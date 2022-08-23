import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TopPLogitsWarper, TopKLogitsWarper, LogitsProcessorList
import pandas as pd
from datasets import load_dataset

import pickle as pkl
import argparse
import os

# set DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_src(split):
    '''generator object that loads articles from disk'''
    dataset = load_dataset("cnn_dailymail", '3.0.0', split=split)
    for article in dataset:
        if article['article'].strip():
            yield article['article']


def load_ref(split):
    '''generator object that loads articles from disk'''
    dataset = load_dataset("cnn_dailymail", '3.0.0', split=split)
    for article in dataset:
        if article['highlights'].strip():
            yield [article['highlights']]


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

            encoder_inputs = tokenizer(src, truncation=True, padding='longest',
                                       max_length=256, return_tensors='pt').to(DEVICE)

            # encode inputs
            encoded = model(**encoder_inputs,
                            output_hidden_states=True, return_dict=True)

            for ref in refs:
                # cannot batch decode because padding  adds dummy probabilites that are hard to get rid off
                decoder_inputs = tokenizer(ref, truncation=True, padding='longest',
                                           max_length=256, return_tensors='pt').to(DEVICE)

                padded = decoder_inputs.input_ids
                # forward pass
                encoder_outputs = (
                    encoded.encoder_last_hidden_state, encoded.encoder_hidden_states)

                forward = model(
                    input_ids=None, encoder_outputs=encoder_outputs, labels=padded, return_dict=True)
                # save
                shift_logits = forward.logits[:, 1:, :].contiguous()
                shift_labels = decoder_inputs.input_ids[:, 1:].contiguous()

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
                    p_num.squeeze().to('cpu'))
                # calculate epsilon
                eps = diff.gather(-1, shift_labels.unsqueeze(-1))
                epsilon_list.append(eps.squeeze().to('cpu'))

                shift_ents = ent

                log_probs.append(log_softmax(
                    shift_logits, shift_labels).to('cpu'))
                tokens.append(decoder_inputs.input_ids.to('cpu'))
                text.append(ref)
                ent_list.append(shift_ents.to('cpu'))

    return {'log_probs': log_probs, 'tokens': tokens, 'text': text, 'entropy': ent_list, 'p': p_nums,
            'epsilon': epsilon_list}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help='output directory')
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('-n', '--num_refs', type=int,
                        default=4, help='input file')
    parser.add_argument('--refs', action='store_true',
                        help='forward pass references, if not set forward pass generations instead')
    parser.add_argument('--top_p', type=float,
                        help='if set to a value 0 < top_p < 1 warp the logits accordingly')
    parser.add_argument('--top_k', type=int,
                        help='perform top_k logits warping')

    cl_args = parser.parse_args()

    output_path = cl_args.output or '../data/forward/news'
    num_refs = cl_args.num_refs

    top_p = cl_args.top_p
    top_k = cl_args.top_k

    # load data
    ref_path = cl_args.input or '../data/generations/news.generations.final.p'
    split = 'test'

    src_iterator = list(load_src(split))[:400]

    #ref_iterator = load_gens(ref_path, 400)
    ref_iterator = load_ref(
        split) if cl_args.refs else load_gens(ref_path, 400)

    model_path = '../../models/bart_cnn/final'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path).to(DEVICE)
    model.eval()

    res = calc_logits(model, tokenizer, src_iterator,
                      ref_iterator, top_p, top_k)

    # save
    filename = 'news_refs_probs' if cl_args.refs else 'news_gens_probs'
    if top_p:
        filename += '_' + str(top_p)
    if top_k:
        filename += '_' + str(top_k)
    filename += '.p'
    output_path = os.path.join(output_path, filename)
    pkl.dump(res, open(output_path, 'wb'))


if __name__ == '__main__':
    main()
