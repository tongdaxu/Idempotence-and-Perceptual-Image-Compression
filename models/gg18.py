import torch
from compressai.models import ScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional, EntropyModel
import numpy as np

class EntropyBottleneckNoQuant(EntropyBottleneck):
    def __init__(self, channels):
        super().__init__(channels)

    def forward(self, x_quant):
        perm = np.arange(len(x_quant.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x_quant.shape))[np.argsort(perm)]
        x_quant = x_quant.permute(*perm).contiguous()
        shape = x_quant.size()
        x_quant = x_quant.reshape(x_quant.size(0), 1, -1)
        likelihood = self._likelihood(x_quant)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return likelihood

class GaussianConditionalNoQuant(GaussianConditional):
    def __init__(self, scale_table):
        super().__init__(scale_table=scale_table)

    def forward(self, x_quant, scales, means):
        likelihood = self._likelihood(x_quant, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return likelihood

class ste_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, x_hat_grad):
        return x_hat_grad.clone()

cvt = {
    "entropy_bottleneck._matrices.0": "entropy_bottleneck._matrix0",
    "entropy_bottleneck._matrices.1": "entropy_bottleneck._matrix1",
    "entropy_bottleneck._matrices.2": "entropy_bottleneck._matrix2",
    "entropy_bottleneck._matrices.3": "entropy_bottleneck._matrix3",
    "entropy_bottleneck._matrices.4": "entropy_bottleneck._matrix4",
    "entropy_bottleneck._biases.0": "entropy_bottleneck._bias0",
    "entropy_bottleneck._biases.1": "entropy_bottleneck._bias1",
    "entropy_bottleneck._biases.2": "entropy_bottleneck._bias2",
    "entropy_bottleneck._biases.3": "entropy_bottleneck._bias3",
    "entropy_bottleneck._biases.4": "entropy_bottleneck._bias4",
    "entropy_bottleneck._factors.0": "entropy_bottleneck._factor0",
    "entropy_bottleneck._factors.1": "entropy_bottleneck._factor1",
    "entropy_bottleneck._factors.2": "entropy_bottleneck._factor2",
    "entropy_bottleneck._factors.3": "entropy_bottleneck._factor3",
    "entropy_bottleneck._factors.4": "entropy_bottleneck._factor4",
}
class ScaleHyperpriorSTE(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneckNoQuant(N)
        self.gaussian_conditional = GaussianConditionalNoQuant(None)

    def quantize(self, inputs, mode):
        return ste_round.apply(inputs)

    def forward(self, x, mode="all"):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        y_hat = self.quantize(y, "round")
        if mode == "enc":
            return { 
                "y_hat": y_hat
            }
        z_hat = self.quantize(z, "round")
        z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_likelihoods = self.gaussian_conditional(y_hat, scales_hat, None)
        x_hat = self.g_s(y_hat)
        return {
            "x_bar": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    def load_state_dict_gg18(self, sd):
        sdkeys = list(sd.keys())
        for key in sdkeys:
            if key not in cvt.keys():
                continue
            sd[cvt[key]] = sd[key]
            del sd[key]
        self.load_state_dict(sd)