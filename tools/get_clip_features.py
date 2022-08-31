# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import torch.nn.functional as F
sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
import detic.modeling.clip as CLIP
from detic.modeling import clip

model, _ = CLIP.load(name='ViT-B/32',
                     use_image_encoder=True,
                     download_root='models')
model.load_state_dict(torch.load('models/clip_vit32.pth', map_location=torch.device('cpu')), strict=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--out_path', default='datasets/metadata/lvis_clip_word_embeddings.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--avg_synonyms', action='store_true')
    parser.add_argument('--use_wn_name', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
                 sorted(data['categories'], key=lambda x: x['id'])]
    if 'synonyms' in data['categories'][0]:
        if args.use_wn_name:
            synonyms = [
                [xx.name() for xx in wordnet.synset(x['synset']).lemmas()] \
                    if x['synset'] != 'stop_sign.n.01' else ['stop_sign'] \
                for x in sorted(data['categories'], key=lambda x: x['id'])]
        else:
            synonyms = [x['synonyms'] for x in \
                        sorted(data['categories'], key=lambda x: x['id'])]
    else:
        synonyms = []
    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
                              for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
                              for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
          sum(len(x) for x in sentences_synonyms))

    with torch.no_grad():
        print('Loading CLIP')
        # model, preprocess = clip.load(args.clip_model, device=device)
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        tokens = clip.get_tokens(sentences)
        embeddings = torch.stack([model.token_embedding.weight[tk].mean(0) for tk in tokens])
        # embeddings = F.normalize(embeddings, dim=-1, p=2)
    if args.out_path != '':
        print('saveing to', args.out_path)
        np.save(open(args.out_path, 'wb'), embeddings.cpu().numpy())
    # import pdb; pdb.set_trace()
