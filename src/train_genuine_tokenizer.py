"""Train CE-only, genuine-tokenizer PAATRA baselines on WikiText-103."""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

def device():
    if torch.cuda.is_available(): return 'cuda', torch.float16
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): return 'mps', torch.bfloat16
    return 'cpu', torch.float32

# NOTE (kept to reproduce the reported runs): this formula counts the embedding matrix
# twice and omits position embeddings, but GPT2LMHeadModel ties input/output embeddings.
# The resulting models therefore have 57.4M/51.4M/41.9M parameters (10K/20K/50K vocab),
# not the ~64M target. See the paper's genuine-tokenizer appendix.
def params(vocab, d, layers=8):
    return vocab*d + layers*(12*d*d + 13*d) + vocab*d

def matched_dim(vocab, target=64_000_000):
    choices = range(256, 1025, 8)
    return min(choices, key=lambda d: abs(params(vocab, d)-target))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--num-steps', type=int, default=15000); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir); (out/'students').mkdir(parents=True, exist_ok=True); (out/'logs').mkdir(exist_ok=True)
    log_path = out/'logs'/'train.log'
    def log(s):
        print(s, flush=True)
        with log_path.open('a') as f: f.write(s+'\n')
    tok = Tokenizer.from_file(a.tokenizer); vocab = tok.get_vocab_size(); d = matched_dim(vocab); N = params(vocab, d)
    dev, dtype = device(); log(f'tokenizer={a.tokenizer} vocab={vocab} d={d} params={N} device={dev}')
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
    texts = [r['text'] for r in ds if isinstance(r['text'], str) and len(r['text'].strip()) >= 20]
    ids = []
    for text in texts: ids.extend(tok.encode(text).ids)
    seq = 512; n = len(ids)//seq; chunks = torch.tensor(ids[:n*seq], dtype=torch.long).reshape(n, seq)
    cfg = GPT2Config(vocab_size=vocab, n_embd=d, n_layer=8, n_head=8, n_inner=4*d,
                     activation_function='gelu_new', n_positions=seq, resid_pdrop=0.0,
                     embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=None, eos_token_id=None)
    model = GPT2LMHeadModel(cfg).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    model.train(); step=0; losses=[]; run=0.0; start=time.time(); order=torch.arange(n)
    while step < a.num_steps:
        order = order[torch.randperm(n)]
        for j in range(0, n, a.batch_size):
            if step >= a.num_steps: break
            x = chunks[order[j:j+a.batch_size]].to(dev); logits = model(input_ids=x).logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), x[:,1:].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += float(loss.item()); step += 1
            if step % 500 == 0:
                val=run/500; losses.append({'step':step,'loss':val}); log(f'step={step} loss={val:.6f} elapsed_min={(time.time()-start)/60:.1f}'); run=0.0
    bundle={'name':out.name,'state_dict':model.state_dict(),'config':{'n_embd':d,'n_layer':8,'n_head':8,'vocab_K':vocab},
            'total_params':sum(x.numel() for x in model.parameters()),'vocab_size':vocab,'num_steps':a.num_steps,
            'losses':losses,'tokenizer_path':a.tokenizer,'hyperparams':{'lr':a.lr,'seed':a.seed,'batch_size':a.batch_size,'objective':'CE-only genuine-tokenizer baseline'}}
    torch.save(bundle, out/'students'/f'{out.name}.pt'); log(f'COMPLETE steps={step} params={bundle["total_params"]} chunks={n}')

if __name__ == '__main__': main()
