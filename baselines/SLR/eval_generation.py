import argparse
import os.path as osp
import numpy as np
import json

import chainer
from chainer import Variable, cuda, serializers

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind, beam_search
from models.base import VisualEncoder, LanguageEncoder, LanguageEncoderAttn
from models.Listener import CcaEmbedding
from models.LanguageModel import vis_combine, LanguageModel
from misc.eval_utils import compute_margin_loss, computeLosses, language_eval
import config

def eval_all(params):
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'],'model', params['dataset']+'_'+params['splitBy'])
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats'] = 'old'+params['image_feats']
        params['ann_feats'] = 'old'+params['ann_feats']
        params['id'] = 'old'+params['id']
        
    loader = DataLoader(params)
    
    featsOpt = {'ann':osp.join(target_save_dir, params['ann_feats']),
                'img':osp.join(target_save_dir, params['image_feats'])}
    loader.loadFeats(featsOpt) 
    chainer.config.train = False
    chainer.config.enable_backprop = False
    
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    
    if 'attention' in params['id']:
        print('attn')
        le = LanguageEncoderAttn(len(loader.ix_to_word)).to_gpu(gpu_id)
    else:
        le = LanguageEncoder(len(loader.ix_to_word)).to_gpu(gpu_id)
    ve = VisualEncoder().to_gpu(gpu_id)
    cca = CcaEmbedding().to_gpu(gpu_id)
    #lm = LanguageModel(len(loader.ix_to_word), loader.seq_length).to_gpu(gpu_id)
    NUM_ATTRS = 26 + 38  # (color, location, action) + (count)
    lm = LanguageModel(len(loader.ix_to_word), loader.seq_length, attr_size=NUM_ATTRS).to_gpu(gpu_id)

    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"ve.h5"), ve)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"le.h5"), le)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"cca.h5"), cca)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"lm.h5"), lm)
    
    predictions = []
    beam_all_results = []
    while True:
        data = loader.getTestBatch(params['split'], params)
        ref_ids = data['ref_ids']
        
        lang_last_ind = calc_max_ind(data['seqz'])
        feats = Variable(xp.array(data['feats'], dtype=xp.float32))
        vis_enc_feats, attrs = ve(feats)
        lang_enc_feats = vis_enc_feats ##fake
        _, vis_emb_feats = cca(vis_enc_feats, lang_enc_feats)
        vis_feats = vis_combine(vis_enc_feats, vis_emb_feats)
        
        if params['beam_width']==1:
            results = lm.max_sample(vis_feats, attrs=attrs)
        else:
            beam_results = beam_search(lm, vis_feats, params['beam_width'], batch_attrs=attrs)
            results = [result[0]['sent'] for result in beam_results]
            ppls = [result[0]['ppl'] for result in beam_results]
            
        for i, result in enumerate(results):
            gen_sentence= ' '.join([loader.ix_to_word[str(w)] for w in result])
            if params['beam_width']==1:
                print(gen_sentence)
            else:
                print(gen_sentence, 'ppl : ', ppls[i])
            entry = {'ref_id':ref_ids[i], 'sent':gen_sentence}
            predictions.append(entry)
            if params['beam_width']>1:
                beam_all_results.append({'ref_id':ref_ids[i], 'beam':beam_results[i]})
        print('evaluating validation performance... {}/{}'.format(data['bounds']['it_pos_now'], data['bounds']['it_max']))
        
        if data['bounds']['wrapped']:
            print('validation finished!')
            break
    lang_stats = language_eval(predictions, params['split'], params)
    print(lang_stats)
    with open('result/'+params['dataset']+params["split"]+'_'+params['id']+params['id2']+str(params['beam_width'])+'raw.json','w') as f:
        json.dump(predictions, f)
    
    with open('result/'+params['dataset']+params["split"]+'_'+params['id']+str(params['beam_width'])+'.json','w') as f:
        json.dump(lang_stats, f)
    
    if params['beam_width']>1:
        with open(target_save_dir+params["split"]+'_'+params['id']+str(params['beam_width'])+'.json','w') as f:
            json.dump(beam_all_results, f)
        
                    
# python eval_generation.py -id 96 -beam 3
if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    eval_all(params)