import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)

ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
texts = [r['text'] for r in ds if isinstance(r['text'], str) and len(r['text'].strip()) >= 20]

def iterator():
    yield from texts

def train_one(kind, k):
    specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    if kind == 'bpe':
        tok = Tokenizer(BPE(unk_token='<unk>'))
        trainer = BpeTrainer(vocab_size=k, min_frequency=2, special_tokens=specials,
                             initial_alphabet=ByteLevel.alphabet())
    else:
        tok = Tokenizer(Unigram())
        trainer = UnigramTrainer(vocab_size=k, special_tokens=specials, unk_token='<unk>')
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train_from_iterator(iterator(), trainer=trainer, length=len(texts))
    path = OUT / f'{kind}_{k}.json'
    tok.save(str(path))
    sample = texts[:1000]
    lengths = [len(tok.encode(t).ids) for t in sample]
    unknown = sum(tok.token_to_id('<unk>') in tok.encode(t).ids for t in sample)
    return {'kind': kind, 'requested_vocab': k, 'actual_vocab': tok.get_vocab_size(),
            'documents': len(sample), 'mean_tokens': sum(lengths) / len(lengths),
            'unk_document_rate': unknown / len(sample), 'path': str(path)}

rows = []
for kind in ('bpe', 'unigram'):
    for k in (10000, 20000, 50000):
        rows.append(train_one(kind, k))

(OUT / 'tokenizer_stats.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
