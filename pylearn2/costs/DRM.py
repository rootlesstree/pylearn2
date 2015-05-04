from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
import theano.tensor as T
#from pylearn2.utils import sharedX
from theano import shared

class DRMCost(DefaultDataSpecsMixin, Cost):
    supervised = False
    
    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        p =0.001        
        X = data
        loss_kl = shared(0,'loss_kl')
        for i in range(model.num_levels()):
            p_hat = model.reconstruct_to_level(X,i+1).mean(axis=0)
            kl = p * T.log(p / p_hat) + (1- p) * T.log((1-p)/(1-p_hat))
            loss_kl = loss_kl + kl.sum()

        X_hat = model.reconstruct(X)
        loss_data = ((X - X_hat)**2).sum(axis=1)
        Weights = model.get_all_weights()
        loss_weight = shared(0,'loss_weight')
        
        for w in Weights:
            loss_weight = loss_weight + T.sum(T.sum(w**2,axis = 0))
        
        
#        print(loss_weight.ndim)
#        print(loss_kl.ndim)
#        print(str(T.iscalar(loss_data.mean())))
#        print(str(T.iscalar(loss_kl)))
#        print(str(T.iscalar(loss_weight)))
        #loss = -(X * T.log(X_hat) + (1 - X) * T.log(1 - X_hat)).sum(axis=1)
        loss = loss_data.mean() #+ 0.5*loss_kl + 0.01*loss_weight
        #loss = 0.5*loss_kl
        return loss





