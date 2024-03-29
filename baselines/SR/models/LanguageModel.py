import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
from misc.utils import softmax_sample
from chainer import variable


def vis_combine(vis_enc, vis_emb, init_norm=20):
    return F.concat([vis_enc, F.dropout(vis_emb*init_norm, ratio=0.25)], axis=1)

class LanguageModel(chainer.Chain):
    def __init__(self,vocab_size, seq_length, attr_size):
        super(LanguageModel, self).__init__(
            word_emb = L.EmbedID(vocab_size+2, 512),
            LSTM = AttrLSTM(512+512, 512, 512, 0.5, attr_size),
            out = L.Linear(512, vocab_size+1),
        )
        self.vocab_size = vocab_size
        self.seq_length = seq_length
            
    #def LSTM_initialize(self):
    #    self.LSTM.reset_state()
    
    def forward(self, feat, w, i, attrs):
        w = self.word_emb(w)
        if i==0:
            _, h = self.LSTM(vis=feat, sos=w, attrs=attrs)
        else:
            _, h = self.LSTM(vis=feat, word=w, attrs=attrs)
        return self.out(h)
            
    def __call__(self, vis_feats, seqz, lang_last_ind, attrs):
        seqz = seqz.data
        xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        #self.LSTM_initialize()
        log_probs = []
        for i in range(max(lang_last_ind)+1):
            if i==0:
                mask = xp.ones(batch_size, dtype=xp.float32)
                sos = Variable(xp.ones(batch_size,dtype=xp.int32)*(self.vocab_size+1))
                sos = self.word_emb(sos)
                _, h = self.LSTM(vis=vis_feats, sos=sos, attrs=attrs)
            else:
                mask = xp.where(seqz[:, i-1]!=0,1,0)
                w = self.word_emb(Variable(seqz[:, i-1]))
                _, h = self.LSTM(vis=vis_feats, word=w, attrs=attrs)
            h = self.out(h)
            logsoft = (F.log_softmax(h)*mask.reshape(batch_size, 1).repeat(h.data.shape[1], axis=1))[np.arange(batch_size), seqz[:,i]]
                
            log_probs.append(logsoft.reshape(1,batch_size)) 
                
        return F.concat(log_probs, axis=0) 
    
    def sample(self, vis_feats, attrs, temperature=1, stochastic=True):
        xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        
        output = xp.zeros((batch_size, self.seq_length), dtype=xp.int32)
        log_probs = [] 
        mask = xp.ones(batch_size)
        
        with chainer.using_config('train', False):
            for i in range(self.seq_length):
                if i==0:
                    sos = self.word_emb(Variable(xp.ones(batch_size,dtype=xp.int32)*(self.vocab_size+1)))
                    _, h = self.LSTM(vis=vis_feats, sos=sos, attrs=attrs)
                else:
                    mask_ = xp.where(w!=0,1,0)
                    mask *= mask_
                    if mask.sum()==0:
                        break
                    w = self.word_emb(Variable(w))
                    _, h = self.LSTM(vis=vis_feats, word=w, attrs=attrs)
                h = self.out(h)
                logsoft = F.log_softmax(h)*mask.reshape(batch_size, 1).repeat(h.data.shape[1], axis=1)# if input==eos then mask

                if stochastic:
                    prob_prev = F.exp(logsoft/temperature)
                    prob_prev /= F.broadcast_to(F.sum(prob_prev, axis=1, keepdims=True), prob_prev.shape)
                    w = softmax_sample(prob_prev)
                else:
                    w = xp.argmax(logsoft.data, axis=1)
                output[:, i] = w
                log_probs.append(logsoft[np.arange(batch_size), w].reshape(1,batch_size))
        return output, F.concat(log_probs, axis=0)

    def max_sample(self, vis_feats, attrs):
        xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        
        output = xp.zeros((batch_size, self.seq_length), dtype=xp.int32)
        mask = xp.ones(batch_size)
        for i in range(self.seq_length):
            if i==0:
                sos = self.word_emb(Variable(xp.ones(batch_size,dtype=xp.int32)*(self.vocab_size+1)))
                _, h = self.LSTM(vis=vis_feats, sos=sos, attrs=attrs)
            else:
                mask_ = xp.where(output[:,i-1]!=0,1,0)
                mask *= mask_
                if mask.sum()==0:
                    break
                w = self.word_emb(Variable(output[:,i-1]))
                _, h = self.LSTM(word=w, attrs=attrs)
            h = self.out(h)
            output[:,i] = xp.argmax(h.data[:,:-1], axis=1)
            
        result = []
        for out in output:
            for i, w in enumerate(out):
                if w==0:
                    result.append(out[:i])
                    break
                
        return result

class AttrLSTM(chainer.Chain):

    def __init__(self, vis_size, word_size, rnn_size, dropout_ratio, attr_size):
        super(AttrLSTM, self).__init__(
            # vis2g=L.Linear(vis_size, rnn_size),
            # h2g=L.Linear(rnn_size, rnn_size, nobias=True),
            # w2g=L.Linear(word_size, rnn_size, nobias=True),
            lstm=L.LSTM(word_size + vis_size, rnn_size),
            ix=L.Linear(word_size, rnn_size, nobias=True),
            ih=L.Linear(rnn_size, rnn_size, nobias=True),
            iv=L.Linear(vis_size, rnn_size, nobias=False),

            fx=L.Linear(word_size, rnn_size, nobias=True),
            fh=L.Linear(rnn_size, rnn_size, nobias=True),
            fv=L.Linear(vis_size, rnn_size, nobias=False),

            ox=L.Linear(word_size, rnn_size, nobias=True),
            oh=L.Linear(rnn_size, rnn_size, nobias=True),
            ov=L.Linear(vis_size, rnn_size, nobias=False),

            cx=L.Linear(word_size, rnn_size, nobias=True),
            ch=L.Linear(rnn_size, rnn_size, nobias=True),
            cv=L.Linear(vis_size, rnn_size, nobias=False),
        )
        self.dropout_ratio = dropout_ratio
        self.hidden_size = rnn_size


        self.h, self.c = None, None

    def reset_state(self, x):
        #self.lstm.reset_state()
        with chainer.using_device(self.device):
            self.c = variable.Variable(self.xp.zeros((x.shape[0], self.hidden_size), dtype=x.dtype))
            self.h = variable.Variable(self.xp.zeros((x.shape[0], self.hidden_size), dtype=x.dtype))

    def __call__(self, vis=None, sos=None, word=None, attrs=None):

        if sos is not None:
            self.reset_state(sos)

        x_t = sos if sos is not None else word

        i_t = F.sigmoid(self.ix(x_t) + self.ih(self.h) + self.iv(vis))
        f_t = F.sigmoid(self.fx(x_t) + self.fh(self.h) + self.fv(vis))
        o_t = F.sigmoid(self.ox(x_t) + self.oh(self.h) + self.ov(vis))

        self.c = f_t * self.c + i_t * F.tanh(self.cx(x_t) + self.ch(self.h) + self.cv(vis))
        #F.tanh(self.cx(x_t) + self.ch(self.h) + self.ca(attrs) + self.cv(vis))

        #self.c = f_t * self.c + i_t * c_tilde_t
        self.h = o_t * F.tanh(self.c)

        #
        # if sos is not None:
        #     input_emb = F.concat([vis, sos], axis=1)
        #     g = self.vis2g(vis) + self.w2g(sos)
        #     h = F.dropout(self.lstm(input_emb), ratio=self.dropout_ratio)
        #
        # else:
        #     word = F.dropout(word, ratio=self.dropout_ratio)
        #     input_emb = F.concat([vis, word], axis=1)
        #     g = F.sigmoid(self.w2g(word) + self.vis2g(vis) + self.h2g(self.lstm.h))
        #     h = F.dropout(self.lstm(input_emb), ratio=self.dropout_ratio)

        #s_t = F.dropout(g * F.tanh(self.lstm.c), ratio=self.dropout_ratio)

        return self.c, self.h


class MyLSTM(chainer.Chain):
    
    def __init__(self, vis_size, word_size, rnn_size, dropout_ratio):
         super(MyLSTM, self).__init__(
                vis2g = L.Linear(vis_size, rnn_size),
                h2g = L.Linear(rnn_size, rnn_size, nobias = True),
                w2g = L.Linear(word_size, rnn_size, nobias = True),
                lstm = L.LSTM(word_size+vis_size, rnn_size),
         )
         self.dropout_ratio = dropout_ratio
    
    def reset_state(self):
        self.lstm.reset_state()
        
    def __call__(self, vis = None, sos = None, word = None):
        
        if sos is not None:
            input_emb = F.concat([vis, sos], axis=1)
            g = self.vis2g(vis)+self.w2g(sos)
            h = F.dropout(self.lstm(input_emb), ratio = self.dropout_ratio)
            
        else:
            word = F.dropout(word, ratio = self.dropout_ratio)
            input_emb = F.concat([vis, word], axis=1)
            g = F.sigmoid(self.w2g(word) + self.vis2g(vis) + self.h2g(self.lstm.h))
            h = F.dropout(self.lstm(input_emb), ratio = self.dropout_ratio)
            
        s_t = F.dropout(g * F.tanh(self.lstm.c), ratio=self.dropout_ratio)
    
        return s_t, h
    
'''
class MyLSTM(chainer.Chain):
    
    def __init__(self, vis_size, word_size, rnn_size, dropout_ratio):
         super(MyLSTM, self).__init__(
                vis2h = L.Linear(vis_size, 4*rnn_size),
                sos2h = L.Linear(word_size, 4*rnn_size, nobias = True),
                af_LSTM = L.LSTM(word_size, rnn_size),
                h2h = L.Linear(rnn_size, rnn_size),
                w2h = L.Linear(word_size, rnn_size, nobias = True),
         )
         self.dropout_ratio = dropout_ratio
    
    def reset_state(self):
        self.af_LSTM.reset_state()
        
    def __call__(self, vis = None, sos = None, word = None):
        if sos is not None:
            h = self.vis2h(vis)+ self.sos2h(sos)
            a, i, o, g = F.split_axis(h, 4, axis = 1)
            a = F.tanh(a)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            g = F.sigmoid(g)
            c = a * i 
            h = F.dropout(o *F.tanh(c), ratio = self.dropout_ratio)
            
            self.af_LSTM.set_state(c, h)
            
        else:
            word_emb = F.dropout(word, ratio = self.dropout_ratio)
            g = F.sigmoid(self.w2h(word_emb) + self.h2h(self.af_LSTM.h))
            h = F.dropout(self.af_LSTM(word_emb), ratio = self.dropout_ratio)
            
        s_t = F.dropout(g * F.tanh(self.af_LSTM.c), ratio=self.dropout_ratio)
    
        return s_t, h

'''