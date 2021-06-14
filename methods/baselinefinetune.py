import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from numpy.polynomial import Polynomial

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func, dataset, tc, n_way, n_support, loss_type = "dist"):
        super(BaselineFinetune, self).__init__( model_func, n_way, n_support)
        self.loss_type = loss_type
        self.dataset = dataset
        self.pos_prob = tc.mean()
        self.tc = torch.cuda.FloatTensor(tc)
        
    def set_forward(self,x,labels,num_classes,is_feature = True):
        return self.set_forward_adaptation(x,labels,num_classes,is_feature); #Baseline always do adaptation
 
    def set_forward_adaptation(self,x,labels,num_classes,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        model = linear_clf
        usebox = True
        if usebox:
            model = backbone.BoxEmbs(self.feat_dim, self.tc.shape[0], labels)
        model = model.cuda()
        set_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        img_loss = nn.CrossEntropyLoss().cuda()
        log1mexp_inv = Polynomial.fit([-i/1000 for i in range(1,2000)],[1/np.log(1-np.exp(-i/1000)) for i in range(1,2000)],2)
        log1mexp = lambda x: 1/log1mexp_inv(x)
        def tc_loss(preds):
            mask = torch.rand(preds.shape) < self.pos_prob
            preds[~self.tc] = (log1mexp(preds) * mask)[~self.tc]
            return -preds.mean()
        support_size = self.n_way * self.n_support
        batch_size = support_size
        scores_eval = []
        for epoch in range(301):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                if usebox:
                    img_scores, tc_scores = model(z_batch)
                else:
                    img_scores = model(z_batch)
                loss = img_loss(img_scores, y_batch)
                if usebox:
                    loss += tc_loss(tc_scores)
                loss.backward()
                set_optimizer.step()
            if epoch %100 ==0 and epoch !=0:
                scores_eval.append(model(z_query))
        return scores_eval


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        

