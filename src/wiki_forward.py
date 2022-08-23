import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TopPLogitsWarper, TopKLogitsWarper, LogitsProcessorList
import pandas as pd
import re

import pickle as pkl
import argparse
import os

# set DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BOS = '<|startoftext|>'


def wiki_loader(path, tgt_len=128, part='ref'):
    if part != 'ref':
        raise ValueError(
            'for this dataset, only reference loader is available')

    def data_loader(path):
        '''generator object for memory efficient file reading'''
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    yield clean(line.strip())

    def batcher(lines, tgt_len):
        full = ' '.join(lines)
        toks = full.split()
        return ([' '.join(toks[i: i + tgt_len])]
                for i in range(0, len(toks), tgt_len))

    def clean(string):
        '''cleans most of the word tonkenization from wikitext'''

        # punctuation and closing parenthesis
        punctuation = re.compile(r'\s([\.\,\;\:\'\!\?\-\_\]\}\)\`])')
        quotes = re.compile(r'((?:^|\s)")\s([^"]+?)\s("(?:\s|$))')  # quotes
        ats = re.compile(r'(\s\@)(.)(\@\s)')  # wierd @ symbols
        opar = re.compile(r'([\(\[\{])\s')  # opening parenthesis

        replacements = [
            (quotes, r'\1\2\3'),
            (punctuation, r'\1'),
            (ats, r'\2'),
            (opar, r'\1')
        ]

        for pattern, new in replacements:
            string = re.sub(pattern, new, string)

        return string

    return batcher(list(data_loader(path)), tgt_len)


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
    ent = []
    for src, refs in zip(src_iterator, ref_iterator):
        with torch.no_grad():

            for ref in refs:

                combined = src + ref
                # cannot batch decode because padding  adds dummy probabilites that are hard to get rid off
                decoder_inputs = tokenizer(combined, truncation=True, padding='longest',
                                           max_length=512, return_tensors='pt').to(DEVICE)

                src_encoding = tokenizer(src, truncation=True, padding='longest',
                                         max_length=512, return_tensors='pt').to(DEVICE)

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

                # save
                log_probs.append({t: log_softmax(
                    shift_logits, shift_labels, temp=t)[:, (src_encoding.input_ids.shape[-1] - 1):].to('cpu') for t in TEMPS})
                tokens.append(decoder_inputs.input_ids.to('cpu'))
                text.append(ref)
                ent.append(entropy(shift_logits)[
                           :, (src_encoding.input_ids.shape[-1] - 1):].to('cpu'))

    return {'log_probs': log_probs, 'tokens': tokens, 'text': text, 'entropy': ent}


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

    output_path = cl_args.output or './'
    num_refs = cl_args.num_refs

    top_p = cl_args.top_p
    top_k = cl_args.top_k

    # load data
    ref_path = '../data/datasets/wikitext-103-raw/wiki.test.raw'
    gen_path = '../data/generations/wiki_medium.generations.final.p'
    tgt_len = 512

    src_iterator = (BOS for _ in range(400))

    # ref_iterator = load_gens(gen_path, 400)
    ref_iterator = wiki_loader(
        ref_path, tgt_len) if cl_args.refs else load_gens(gen_path, 400)

    # enter path to finetuned gpt2 instance
    model_path = '../../models/gpt2_wiki_medium/'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    # init pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    res = calc_logits(model, tokenizer, src_iterator,
                      ref_iterator, top_p, top_k)

    # save
    filename = 'wiki_refs_probs' if cl_args.refs else 'wiki_gens_probs'
    # filename = 'wiki_gens_probs.p'
    if top_p:
        filename += '_' + str(top_p)
    if top_k:
        filename += '_' + str(top_k)
    filename += '.p'
    output_path = os.path.join(output_path, filename)
    pkl.dump(res, open(output_path, 'wb'))


if __name__ == '__main__':
    main()
