import pandas as pd
import numpy as np
df = pd.read_json('/home/ecao/Documents/DL/bert_nli-master/datasets/train.jsonl', lines=True)
sent_pairs = df[['context','hypothesis']].values.tolist()

from bert_nli import BertNLIModel

bert_type = 'bert-large'
model = BertNLIModel('output/{}.state_dict'.format(bert_type), bert_type=bert_type)

features, labels, probs = model(sent_pairs)

from numpy import savez_compressed
savez_compressed('/home/ecao/Documents/DL/bert_nli-master/datasets/features.npz', features)
