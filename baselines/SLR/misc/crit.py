import chainer.functions as F
from chainer import cuda
from chainer import Variable

def emb_crits(emb_flows, margin_same_cat, margin_wrong_cat, vlamda=1, llamda=1):
    xp = cuda.get_array_module(emb_flows['vis'][0])
    batch_size = emb_flows['vis'][0].shape[0]
    # Checked!
    zeros = Variable(xp.zeros(batch_size, dtype=xp.float32))
    vis_margin = margin_same_cat * (emb_flows["c_oi"] == emb_flows["c_oj"]) + margin_wrong_cat * (emb_flows["c_oi"]!=emb_flows["c_oj"])
    vis_loss = F.mean(F.maximum(zeros, vis_margin - emb_flows['d_ri_oj'] + emb_flows['d_ri_oi']))

    lang_margin = margin_same_cat * (emb_flows["c_ri"]==emb_flows["c_rk"]) + margin_wrong_cat * (emb_flows["c_ri"]!=emb_flows["c_rk"])
    lang_loss = F.mean(F.maximum(zeros, lang_margin - emb_flows['d_rk_oi'] + emb_flows['d_ri_oi']))

    return vlamda*vis_loss + llamda*lang_loss
    
def lm_crits(lm_flows, num_labels, margin_same_cat, margin_wrong_cat, vlamda=1, langWeight=1):
    # CHECKED
    xp = cuda.get_array_module(lm_flows['T'])
    ## language loss
    n = 0
    lang_loss = 0
    Tprob = lm_flows['T']
    lang_num = num_labels['T']
    lang_loss -= F.sum(Tprob)/(sum(lang_num)+len(lang_num)) # + EOS
    if vlamda==0:
        return lang_loss
    
    def triplet_loss(flow, num_label, cats):
        pairGenP = flow[0]
        unpairGenP = flow[1]
        pairGenCats = cats[0]
        unpairCats = cats[1]
        zeros = Variable(xp.zeros(pairGenP.shape[1], dtype=xp.float32))
        pairSentProbs = F.sum(pairGenP,axis=0)/(num_label+1)
        unpairSentProbs = F.sum(unpairGenP,axis=0)/(num_label+1)
        trip_loss = F.mean(F.maximum(zeros, margin_same_cat * (pairGenCats==unpairCats) +
                                     margin_wrong_cat * (pairGenCats!=unpairCats) + unpairSentProbs-pairSentProbs))
        return trip_loss
    
    #vloss = triplet_loss(lm_flows['visF'], xp.array(num_labels['T']), lm_flows["visF_cats"])
    num_label = xp.array(num_labels['T'])
    p_ri_ok = F.sum(lm_flows['P(ri|ok)'],axis=0)/(num_label+1)
    p_ri_oi = F.sum(lm_flows['P(ri|oi)'],axis=0)/(num_label+1)
    zeros = Variable(xp.zeros(lm_flows['P(ri|oi)'].shape[1], dtype=xp.float32))
    c_oi = lm_flows['c_oi']
    c_ok = lm_flows['c_ok']
    margin_vision = margin_same_cat * (c_oi==c_ok) + margin_wrong_cat * (c_oi!=c_ok)
    vloss = F.mean(F.maximum(zeros, margin_vision + p_ri_ok - p_ri_oi))
    #lloss = triplet_loss(lm_flows['langF'], xp.array(num_labels['F']))
    #print(lang_loss, vloss, lloss)
    return langWeight*lang_loss + vlamda*vloss #+llamda*lloss