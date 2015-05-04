from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.models.mlp import PretrainedLayer
import theano
from theano import tensor
import copy
import numpy
import theano.tensor as T
from pylearn2.utils import sharedX

class DeepEncoder(Model):
    def __init__(self, layers):
        super(DeepEncoder, self).__init__()

        # Some parameter initialization using *args and **kwargs
        # ...]
        self._layers = [l for l in layers]
        self._encode_layers = [copy.deepcopy(l) for l in layers]
        self._decode_layers = [copy.deepcopy(l) for l in layers]
        #[p for l in self._layers for p in l._params]        
        self._params = []
        for l in self._encode_layers:
            for p in l._params:
                self._params.append(p)
        for l in self._decode_layers:
            for p in l._params:
                self._params.append(p)
        self.input_space = VectorSpace(layers[0].nvis)
        self.output_space = VectorSpace(layers[-1].nhid)

    def num_levels(self):
        return len(self._layers)

    def encode_to_level(self, inputs, level):
        if(level > len(self._layers)):
            raise ValueError("level is larger than what model has")

        if isinstance(inputs, tensor.Variable):
            next_input = inputs
            for l in self._encode_layers[0:level]:
                character = [e.name for e in l._params]
                W = l._params[character.index('W')]
                b = l._params[character.index('bias_hid')]
                next_input = T.nnet.sigmoid(T.dot(next_input,W) + b)
            return next_input
        else:
            return [self.encode(v) for v in inputs]

    def decode_to_level(self, hiddens, level):
        if(level > len(self._layers)):
            raise ValueError("level is larger than what model has")

        if isinstance(hiddens, tensor.Variable):
            next_input = hiddens
            for l in reversed(self._decode_layers[0:level]):
                character = [e.name for e in l._params]
                W = l._params[character.index('W')]
                c = l._params[character.index('bias_vis')]
                next_input = T.nnet.sigmoid(T.dot(next_input,W.T) + c)
            return next_input
        else:
            return [self.decode(v) for v in hiddens]





    def encode(self, inputs):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        encoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after encoding.
        """
        if isinstance(inputs, tensor.Variable):
            next_input = inputs
            for l in self._encode_layers:
                character = [e.name for e in l._params]
                W = l._params[character.index('W')]
                b = l._params[character.index('bias_hid')]
                next_input = T.nnet.sigmoid(T.dot(next_input,W) + b)
            return next_input
        else:
            return [self.encode(v) for v in inputs]

    def decode(self, hiddens):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        hiddens : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        decoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after decoding.
        """
        if isinstance(hiddens, tensor.Variable):
            next_input = hiddens
            for l in reversed(self._decode_layers):
                character = [e.name for e in l._params]
                W = l._params[character.index('W')]
                c = l._params[character.index('bias_vis')]
                next_input = T.nnet.sigmoid(T.dot(next_input,W.T) + c)
            return next_input
        else:
            return [self.decode(v) for v in hiddens]
    def reconstruct_to_level(self, X, level):
        return self.decode_to_level(self.encode_to_level(X,level),level)
    def reconstruct(self, X):
        return self.decode(self.encode(X))

    def reconstruct_error(self,X):
        X_hat = self.reconstruct(X)
        loss_data = ((X - X_hat)**2).sum(axis=1)
        return loss_data

    def get_all_weights(self):
        return_weight = []
        for l in self._encode_layers:
            character = [e.name for e in l._params]
            W = l._params[character.index('W')]
            return_weight.append(W)
        for l in self._decode_layers:
            character = [e.name for e in l._params]
            W = l._params[character.index('W')]
            return_weight.append(W)
        return return_weight

    def get_activation_value(self, X):
        if isinstance(inputs, tensor.Variable):
            return_activation = []
            next_input = inputs
            for l in self._layers:
                character = [e.name for e in l._params]
                W = l._params[character.index('W')]
                b = l._params[character.index('bias_hid')]
                next_input = T.nnet.sigmoid(T.dot(next_input,W) + b)
                return_activation.append(next_input)
            return return_activation
        else:
            return [self.encode(v) for v in inputs]
       