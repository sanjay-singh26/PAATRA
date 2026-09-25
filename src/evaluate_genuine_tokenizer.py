import argparse, json, math
from pathlib import Path
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2Config

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--tokenizer',required=True); p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=8); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); tok=Tokenizer.from_file(a.tokenizer); ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='test')
    texts=[r['text'] for r in ds if isinstance(r['text'],str) and len(r['text'].strip())>=1]; ids=[]; nbytes=0
    for text in texts: ids.extend(tok.encode(text).ids); nbytes += len(text.encode('utf-8'))
    seq=ck.get('config',{}).get('seq_len',512); n=len(ids)//seq; x=torch.tensor(ids[:n*seq],dtype=torch.long).reshape(n,seq)
    cfg=GPT2Config(vocab_size=ck['vocab_size'],n_embd=ck['config']['n_embd'],n_layer=ck['config']['n_layer'],n_head=ck['config']['n_head'],n_inner=4*ck['config']['n_embd'],activation_function='gelu_new',n_positions=seq,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,bos_token_id=None,eos_token_id=None)
    model=GPT2LMHeadModel(cfg); model.load_state_dict(ck['state_dict']); model.eval(); total=0.; count=0
    with torch.no_grad():
        for i in range(0,n,a.batch_size):
            z=model(input_ids=x[i:i+a.batch_size]).logits[:,:-1,:]; y=x[i:i+a.batch_size,1:]
            total += torch.nn.functional.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),reduction='sum').item(); count += y.numel()
    row={'checkpoint':str(a.checkpoint),'tokenizer':str(a.tokenizer),'vocab_size':ck['vocab_size'],'params':ck['total_params'],'test_tokens':count,'test_bytes':nbytes,'token_nll':total/count,'bpb':total/math.log(2)/nbytes}
    output_path = Path(a.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(row,indent=2)); print(json.dumps(row,indent=2))
if __name__=='__main__': main()
