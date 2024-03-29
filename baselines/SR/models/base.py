import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda

class VisualEncoder(chainer.Chain):
    def __init__(self, res6=None, res_dim=2048, res6_dim=1000, encoding_size=512, dif_num=5, num_attrs=26):
        initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
        super(VisualEncoder, self).__init__(
            cxt_enc  = L.Linear(res_dim, res6_dim),
            ann_enc = L.Linear (res_dim, res6_dim),
            dif_ann_enc = L.Linear(res_dim, res6_dim),
            joint_enc = L.Linear(res6_dim*3+5*(dif_num+1), encoding_size, initialW=initializer),
            #C = 512, n_attrs = config["n_attrs"], hidden_dim = 512)
            W_s = L.Linear(2048, 512),
            W_a_s = L.Linear(60, 512),
            W_s_to_softmax = L.Linear(512, 1),

            W_p=L.Linear(2048, 512),
            W_a_p=L.Linear(60, 512),
            W_p_to_softmax=L.Linear(512, 1),
            weighted_to_feats = L.Linear(2048*6*6, 2048)
        )
        self.feat_ind = [2048, 2048, 5, 2048, 25, num_attrs]
        if res6 !=None:
            self.cxt_enc = res6.copy()
            self.ann_enc = res6.copy()
            self.dif_ann_enc = res6.copy()

    # def forward(self, V, attr):
    #     """
    #
    #     :param V: of shape [B, H, W, C]
    #     :param attr: of shape [B, n_attrs]
    #     :return: A weighted img tensor [B, H, W, C]
    #     """
    #     V_bar = self.spatial_wise_attention(V, attr) # shape [B,C]
    #     V_tilde = self.channel_wise_attention(V, attr)  # shape [B, H, W]
    #
    #
    #     V_bar = V_bar.unsqueeze(1).unsqueeze(1) # shape [B, 1, 1, C]
    #     V_tilde = V_tilde.unsqueeze(-1) # shape [B, H, W, 1]
    #
    #     return V_tilde * V * V_bar
    def weightedFeatureMap(self, feat_maps, attrs):
        """

        :param feat_maps: B, 6, 6, 2048
        :param attrs: B, 60
        :return:
        """
        # To use the same names as in the paper
        a = attrs
        V = feat_maps
        xp = cuda.cupy

        [B, H, W, C] = V.shape
        S_a = F.tanh(self.W_s(V) + self.W_a_s(a)[xp.newaxis, xp.newaxis, :]) # B,H,W,d
        a_v = F.softmax(self.w_s_to_softmax(S_a.view(B,H*W,-1)), axis=1) # B,H*W,1
        a_v = a_v.view(B, H, W,1)
        V_bar = (a_v*V).view(B,H*W,C).sum(1)

        P_a = F.tanh(self.W_p(V) + self.W_p_s(a)[xp.newaxis, xp.newaxis, :])  # B,H,W,d
        B_v = F.softmax(self.w_s_to_softmax(P_a.view(B, H * W, -1)), axis=1)  # B,H*W,1
        B_v = B_v.view(B, H, W, 1)
        V_tilde = (B_v * V).sum(-1)

        V_bar = V_bar.unsqueeze(1).unsqueeze(1)  # shape [B, 1, 1, C]
        V_tilde = V_tilde.unsqueeze(-1) # shape [B, H, W, 1]

        weighted_map = V_tilde * V * V_bar # [B, H, W, C]

        feats = self.weighted_to_feats(weighted_map.view(B,H*W*C)) # [B, H*W*C] -> [B, 2048]
        return feats

    def __call__(self, feats, feat_maps, attrs, init_norm=20):
        cxt = self.cxt_enc(feats[:, :self.feat_ind[0]])
        #ann = self.ann_enc(feats[:, sum(self.feat_ind[:1]):sum(self.feat_ind[:2])])
        ann = self.weightedFeatureMap(feat_maps, attrs)

        loc = feats[:, sum(self.feat_ind[:2]):sum(self.feat_ind[:3])]
        diff_ann = self.dif_ann_enc(feats[:, sum(self.feat_ind[:3]):sum(self.feat_ind[:4])])
        diff_loc = feats[:, sum(self.feat_ind[:4]):sum(self.feat_ind[:5])]
        
        cxt = F.normalize(cxt)*init_norm
        ann = F.normalize(ann)*init_norm
        loc = F.normalize(loc+1e-15)*init_norm
        diff_ann = F.normalize(diff_ann)*init_norm
        diff_loc = F.normalize(diff_loc+1e-15)*init_norm
        
        J = F.concat([cxt, ann, loc, diff_ann, diff_loc], axis=1)
        J = F.dropout(self.joint_enc(J), ratio=0.25)
        return J, feats[:, sum(self.feat_ind[:5]):]
    
class LanguageEncoderAttn(chainer.Chain):
    def __init__(self,vocab_size):
        super(LanguageEncoderAttn, self).__init__(
            word_emb = L.EmbedID(vocab_size+2, 512),
            LSTM = L.LSTM(512, 512),
            linear1 = L.Linear(512, 512),
            linear2 = L.Linear(512, 1),
            norm = L.BatchNormalization(512, eps=1e-5),
        )
        
    def LSTMForward(self, sents_emb, max_last_ind):
        self.LSTM.reset_state()
        h_list = []
        for i in range(max_last_ind+1):
            h = self.LSTM(sents_emb[:,i])
            h_list.append(h)# length*b*512
        return h_list
    
    def create_word_mask(self, lang_last_ind, xp):
        mask = xp.zeros((len(lang_last_ind), max(lang_last_ind)+1), dtype=xp.float32)
        for i in range(len(lang_last_ind)):
            mask[i,:lang_last_ind[i]+1] = 1
        return mask
    
    def sentence_attention(self, lstm_out, lang_last_ind):
        batch_size = len(lang_last_ind)
        seq_length = max(lang_last_ind)+1
        lstm_out = F.reshape(F.concat(lstm_out, axis = 1), (batch_size*seq_length, -1))
        xp = cuda.get_array_module(lstm_out)
        
        word_mask = self.create_word_mask(lang_last_ind, xp) #b*seq_length
        h = F.dropout(F.relu(self.linear1(lstm_out)), ratio=0.1)
        h = F.reshape(self.linear2(h), (batch_size, seq_length))
        h = h*word_mask+(word_mask*1024-1024) 
        att_softmax = F.softmax(h, axis=1)
        self.attention_result = att_softmax
        lstm_out = F.reshape(lstm_out, (batch_size, seq_length, -1))
        att_mask = F.broadcast_to(F.reshape(att_softmax, (batch_size, seq_length, 1)), lstm_out.shape)  # N x T  x d
        att_mask = att_mask * lstm_out 
        att_mask = F.sum(att_mask, axis = 1)
        return att_mask
    
    def __call__(self, sents, lang_last_ind, attention=True):
        sents_emb = F.dropout(self.word_emb(sents), ratio=0.5)
        sents_emb = self.LSTMForward(sents_emb, max(lang_last_ind))
        sents_emb = self.norm(self.sentence_attention(sents_emb, lang_last_ind))
        return sents_emb
    
class LanguageEncoder(chainer.Chain):
    def __init__(self,vocab_size, num_attrs=60):
        super(LanguageEncoder, self).__init__(
            word_emb = L.EmbedID(vocab_size+2, 512),
            LSTM = L.NStepBiLSTM(n_layers=1,in_size=512, out_size=512), #L.LSTM(512, 512),
            norm = L.BatchNormalization(512, eps=1e-5),
            W_h = L.Linear(512, 512),
            W_a_h = L.Linear(num_attrs, 512),
        )
        
    def LSTMForward(self, sents_emb, max_last_ind):
        self.LSTM.reset_state()
        h_list = []
        for i in range(max_last_ind+1):
            h = self.LSTM(sents_emb[:,i])
            h_list.append(h)# length*b*512
        return h_list
    
    def __call__(self, sents, lang_last_ind, attrs):
        sents_emb = F.dropout(self.word_emb(sents), ratio=0.5)
        sents_emb = self.LSTMForward(sents_emb, max(lang_last_ind))
        sents_emb = self.norm(F.concat([F.reshape(sents_emb[ind][i], (1,-1)) for i, ind in enumerate(lang_last_ind)],axis=0))
        return sents_emb
    
class MetricNet(chainer.Chain):
    def __init__(self):
        initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
        super(MetricNet, self).__init__(
            fc1 = L.Linear(512+512, 512, initialW=initializer),
            norm1 = L.BatchNormalization(512, eps=1e-5),
            fc2 = L.Linear(512, 512, initialW=initializer),
            norm2 = L.BatchNormalization(512, eps=1e-5),
            fc3 = L.Linear(512, 1, initialW=initializer),
            
            vis_norm = L.BatchNormalization(512, eps=1e-5),
        )
        
    def __call__(self, vis, lang):
        joined = F.concat([self.vis_norm(vis), lang], axis=1)
        joined = F.dropout(F.relu(self.norm1(self.fc1(joined))), ratio=0.2)
        joined = F.dropout(F.relu(self.norm2(self.fc2(joined))), ratio=0.2)
        joined = self.fc3(joined)
        return joined