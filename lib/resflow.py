from tkinter import Widget
from turtle import width
import numpy as np
import torch
import torch.nn as nn

import lib.layers as layers
import lib.layers.base as base_layers

from scipy.special import lambertw

ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: nn.ReLU(inplace=b),
}


class ResidualFlow(nn.Module):

    def __init__(
        self,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        factor_out=True,
        quadratic=False,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        classification=False,
        classification_hdim=64,
        n_classes=10,
        block_type='resblock',
        attention=False,
        heads=1
    ):
        super(ResidualFlow, self).__init__()
        self.n_scale = min(len(n_blocks), self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.factor_out = factor_out
        self.quadratic = quadratic
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.vnorms = vnorms
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.kernels = kernels
        self.activation_fn = activation_fn
        self.fc_end = fc_end
        self.fc_idim = fc_idim
        self.n_exact_terms = n_exact_terms
        self.preact = preact
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.first_resblock = first_resblock
        self.learn_p = learn_p
        self.classification = classification
        self.classification_hdim = classification_hdim
        self.n_classes = n_classes
        self.block_type = block_type

        if not self.n_scale > 0:
            raise ValueError(
                'Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size, attention,heads)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

        if self.classification:
            self.build_multiscale_classifier(input_size)

    def _build_net(self, input_size, attention,heads):
        _, c, h, w = input_size
        transforms = []
        _stacked_blocks = StackediResBlocks if self.block_type == 'resblock' else StackedCouplingBlocks
        for i in range(self.n_scale):
            transforms.append(
                _stacked_blocks(
                    initial_size=(c, h, w),
                    idim=self.intermediate_dim,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    quadratic=self.quadratic,
                    actnorm=self.actnorm,
                    fc_actnorm=self.fc_actnorm,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    fc=self.fc,
                    coeff=self.coeff,
                    vnorms=self.vnorms,
                    n_lipschitz_iters=self.n_lipschitz_iters,
                    sn_atol=self.sn_atol,
                    sn_rtol=self.sn_rtol,
                    n_power_series=self.n_power_series,
                    n_dist=self.n_dist,
                    n_samples=self.n_samples,
                    kernels=self.kernels,
                    activation_fn=self.activation_fn,
                    fc_end=self.fc_end,
                    fc_idim=self.fc_idim,
                    n_exact_terms=self.n_exact_terms,
                    preact=self.preact,
                    neumann_grad=self.neumann_grad,
                    grad_in_forward=self.grad_in_forward,
                    first_resblock=self.first_resblock and (i == 0),
                    learn_p=self.learn_p,
                    attention=attention,
                    heads=heads
                )
            )
            c, h, w = c * 2 if self.factor_out else c * 4, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 4**k, h // 2**k, w // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def build_multiscale_classifier(self, input_size):
        n, c, h, w = input_size
        hidden_shapes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2 if self.factor_out else 4
                h //= 2
                w //= 2
            hidden_shapes.append((n, c, h, w))

        classification_heads = []
        for i, hshape in enumerate(hidden_shapes):
            classification_heads.append(
                nn.Sequential(
                    nn.Conv2d(hshape[1], self.classification_hdim, 3, 1, 1),
                    layers.ActNorm2d(self.classification_hdim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
            )
        self.classification_heads = nn.ModuleList(classification_heads)
        self.logit_layer = nn.Linear(
            self.classification_hdim * len(classification_heads), self.n_classes)

    def forward(self, x, logpx=None, inverse=False, classify=False):
        if inverse:
            return self.inverse(x, logpx)
        out = []
        if classify:
            class_outs = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                x, logpx = self.transforms[idx].forward(x, logpx)
            else:
                x = self.transforms[idx].forward(x)
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                x, f = x[:, :d], x[:, d:]
                out.append(f)

            # Handle classification.
            if classify:
                if self.factor_out:
                    class_outs.append(self.classification_heads[idx](f))
                else:
                    class_outs.append(self.classification_heads[idx](x))

        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, logpx)
        if classify:
            h = torch.cat(class_outs, dim=1).squeeze(-1).squeeze(-1)
            logits = self.logit_layer(h)
            return output, logits
        else:
            return output

    def inverse(self, z, logpz=None):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i:i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize)
                  for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].inverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].inverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].inverse(zs[-1], logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].inverse(z_prev, logpz)
                return z_prev, logpz
        else:
            z = z.view(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].inverse(z)
                else:
                    z, logpz = self.transforms[idx].inverse(z, logpz)
            return z if logpz is None else (z, logpz)


class StackediResBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        attention=False,
        heads=1
    ):

        chain = []

        # Parse vnorms
        ps = []
        for p in vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        domains, codomains = ps[:-1], ps[1:]
        assert len(domains) == len(kernels.split('-'))

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _lipschitz_layer(fc):
            return base_layers.get_linear if fc else base_layers.get_conv2d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False, attention=False, heads=1):
            if fc:
                return layers.iResBlock(
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_lipschitz_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        coeff=coeff,
                        domains=domains,
                        codomains=codomains,
                        n_iterations=n_lipschitz_iters,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        sn_atol=sn_atol,
                        sn_rtol=sn_rtol,
                        learn_p=learn_p,
                    ),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )
            else:
                ks = list(map(int, kernels.split('-')))
                if learn_p:
                    _domains = [nn.Parameter(torch.tensor(0.))
                                for _ in range(len(ks))]
                    _codomains = _domains[1:] + [_domains[0]]
                else:
                    _domains = domains
                    _codomains = codomains
                nnet = []
                if not first_resblock and preact:
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(
                    _lipschitz_layer(fc)(
                        initial_size[0], idim, ks[0], 1, ks[0] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[0], codomain=_codomains[0], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm:
                    nnet.append(layers.MovingBatchNorm2d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(
                        _lipschitz_layer(fc)(
                            idim, idim, k, 1, k // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                            domain=_domains[i + 1], codomain=_codomains[i +
                                                                        1], atol=sn_atol, rtol=sn_rtol
                        )
                    )
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))

                if dropout:
                    nnet.append(nn.Dropout2d(dropout, inplace=True))

                if(attention):
                    #nnet.append(L2_Self_Attn(idim,heads=heads))
                    nnet.append(Self_Attn(idim))
                    # nnet.append(ACT_FNS[activation_fn](True))

                nnet.append(
                    _lipschitz_layer(fc)(
                        idim, initial_size[0], ks[-1], 1, ks[-1] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[-1], codomain=_codomains[-1], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm:
                    nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                return layers.iResBlock(
                    nn.Sequential(*nnet),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )

        if init_layer is not None:
            chain.append(init_layer)
        if first_resblock and actnorm:
            chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm:
            chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, h, w = initial_size
            for i in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (
                    i == 0), attention=attention, heads=heads))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, attention=attention,heads=heads))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True,
                                 fc_idim, attention=attention,heads=heads))
                    if actnorm or fc_actnorm:
                        chain.append(_actnorm(initial_size, True))

        super(StackediResBlocks, self).__init__(chain)


class L2_Self_Attn_Old(nn.Module):
    def __init__(self, in_dim, heads=1):
        super(L2_Self_Attn_Old, self).__init__()
        self.chanel_in = in_dim
        self.heads = 1
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # W_Q
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # W_V
        # print(self.query_conv.weight.data.shape,torch.eye(in_dim).shape)
        # self.query_conv.weight.data.copy_(torch.eye(in_dim).view(in_dim,in_dim,1,1))
        # self.value_conv.weight.data.copy_(torch.eye(in_dim).view(in_dim,in_dim,1,1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.one_vec = nn.Parameter(torch.ones(1, requires_grad=False))
        self.softmax = nn.Softmax(dim=-1)

    def row_wise_norm(self, x):
        result = torch.norm(x, dim=2)**2
        return result.view(x.shape[0], x.shape[1], 1)

    def compute_bound(self, N, D, W_Q, W_V):
        e = torch.exp(torch.ones(1)).to(W_Q.device)
        phi = torch.from_numpy(np.real(
            lambertw((N/e).detach().cpu().numpy()))).to(torch.float32).to(W_Q.device)
        phi = phi
        bound = torch.sqrt(torch.tensor(N/D, dtype=torch.float32)) * \
            (4*phi + 1)*torch.norm(W_Q)*torch.norm(W_V)
        return bound

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        #print(m_batchsize, C, width, height)
        x = x
        x_temp = x
        N = width*height
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X N X C
        W_Q1 = self.query_conv.weight
        W_V1 = self.value_conv.weight
        W_Q = W_Q1.view(1, self.query_conv.in_channels,
                        self.query_conv.out_channels).repeat(m_batchsize, 1, 1)
        normed_query = self.row_wise_norm(proj_query)
        one_vec = self.one_vec.view(1, 1).repeat(N, 1).view(
            1, N, 1).repeat(m_batchsize, 1, 1).to(x.device)  # .cuda()
        P_first_term = torch.bmm(normed_query, one_vec.permute(0, 2, 1)).view(m_batchsize, N, N).to(x.device)  # .cuda()
        P_second_term = -2 * torch.bmm(proj_query, proj_query.permute(0, 2, 1)).view(m_batchsize, N, N)
        P_third_term = torch.bmm(one_vec, normed_query.permute(0, 2, 1)).view(m_batchsize, N, N)
        scalar1 = -1/torch.sqrt(torch.tensor(C, dtype=torch.float32)/self.heads)

        P_before_softmax = (scalar1)*(P_first_term +
                                      P_third_term + P_second_term)
        P = self.softmax(P_before_softmax)

        attention = torch.bmm(P, x_temp.view(m_batchsize, N, C))
        scalar2 = 1/torch.sqrt(torch.tensor(C, dtype=torch.float32))

        A_half = self.query_conv(attention.view(
            m_batchsize, -1, width, height)).view(m_batchsize, N, -1)
        attention = (scalar2)*torch.bmm(A_half, W_Q.permute(0, 2, 1))

        out = self.value_conv(attention.view(
            m_batchsize, -1, width, height)).view(m_batchsize, N, -1)
        bound = self.compute_bound(width*height, C, W_Q1, W_V1)
        out /= bound
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma*out + x
        return out

class L2_Self_Attn(nn.Module):
    def __init__(self, in_dim, heads=1):
        super(L2_Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.heads = heads
        self.query_conv_list = list()
        self.value_conv_list = list()
        self.query_conv_list = nn.ModuleList([nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//self.heads, kernel_size=1) for i in range(self.heads)])
        self.value_conv_list = nn.ModuleList([nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//self.heads, kernel_size=1) for i in range(self.heads)])
        # for _ in range(self.heads):
        #     self.query_conv_list.append(nn.Conv2d(
        #     in_channels=in_dim, out_channels=in_dim//self.heads, kernel_size=1))  # W_(Q,h)
        #     self.value_conv_list.append(nn.Conv2d(
        #     in_channels=in_dim, out_channels=in_dim//self.heads, kernel_size=1))  # W_(V,h)
        # print(self.query_conv.weight.data.shape,torch.eye(in_dim).shape)
        # self.query_conv.weight.data.copy_(torch.eye(in_dim).view(in_dim,in_dim,1,1))
        # self.value_conv.weight.data.copy_(torch.eye(in_dim).view(in_dim,in_dim,1,1))
        self.out_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.one_vec = nn.Parameter(torch.ones(1, requires_grad=False))
        self.softmax = nn.Softmax(dim=-1)

    def row_wise_norm(self, x):
        result = torch.norm(x, dim=2)**2
        return result.view(x.shape[0], x.shape[1], 1)
    def compute_fWv(self, x,h):
        B, D, width, height = x.size()
        N = width*height
        dim_heads = D//self.heads
        proj_query = self.query_conv_list[h](x).view(
            B, dim_heads, N).permute(0, 2, 1)  # B X N X D/H
        normed_query = self.row_wise_norm(proj_query) # B X N X 1
        one_vec = torch.ones(B,N,1).to(x.device)
        P_first_term = torch.bmm(normed_query,one_vec.permute(0,2,1)).view(B,N,N)
        P_second_term = (-2)*torch.bmm(proj_query,proj_query.permute(0,2,1)).view(B,N,N)
        P_third_term = torch.bmm(one_vec,normed_query.permute(0,2,1)).view(B,N,N)
        scalar = torch.sqrt(torch.tensor(D/self.heads, dtype=torch.float32))
        P_h = self.softmax((P_first_term + P_second_term + P_third_term)/(-scalar))
        W_Q_h = self.query_conv_list[h].weight.view(1,D,dim_heads).repeat(B,1,1)
        A_h = (torch.bmm(W_Q_h,W_Q_h.permute(0,2,1))/scalar).view(B,D,D)
        #print(A_h.shape)
        f_h = torch.bmm(torch.bmm(P_h,x.view(B,N,D)),A_h).view(B,D,width,height)
        f_h_Wv = self.value_conv_list[h](f_h).view(B,N,dim_heads)

        return f_h_Wv        
    def compute_bound(self, N, D):
        W_Q_list = [query_conv.weight for query_conv in self.query_conv_list]
        W_V_list = [value_conv.weight for value_conv in self.value_conv_list]
        W_O = self.out_conv.weight
        e = torch.exp(torch.ones(1)).to(self.query_conv_list[0].weight.device)
        phi = torch.from_numpy(np.real(
            lambertw((N/e).detach().cpu().numpy()))).to(torch.float32).to(self.query_conv_list[0].weight.device)
        term = 0
        for i in range(self.heads):
            term += (torch.norm(W_Q_list[i])**2)*(torch.norm(W_V_list[i])**2)
        term = torch.sqrt(term)
        bound = torch.sqrt(torch.tensor(N/D, dtype=torch.float32)) *(4*phi + 1)*term*torch.norm(W_O)
        return bound

    def forward(self, x):
        B, D, width, height = x.size()
        N = width*height
        f_list = list()
        for h in range(self.heads):
                f_list.append(self.compute_fWv(x,h))
        out = self.out_conv(torch.cat(tuple(f_list),dim=2).view(B,D,width,height))
        out/= self.compute_bound(N,D)
        out = self.gamma*out.view(B,D,width,height) + x
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,lip=True):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.lip = lip
        if self.lip == True:
            self.norm_value = self.lipschitz_norm()
        self.softmax  = nn.Softmax(dim=-1) #
    def infty_norm(self,x):
        return torch.max(torch.norm(x,dim=1))
    def maximum(self, a, b, c):  
        if (a >= b) and (a >= c):
            largest = a
        elif (b >= a) and (b >= c):
            largest = b
        else:
            largest = c
        return largest
    def lipschitz_norm(self):
        Q = self.query_conv.weight.view(self.query_conv.weight.shape[0],self.query_conv.weight.shape[1])
        K = self.key_conv.weight.view(self.key_conv.weight.shape[0],self.key_conv.weight.shape[1])
        V = self.value_conv.weight.view(self.value_conv.weight.shape[0],self.value_conv.weight.shape[1])
        u,v,w = torch.norm(Q), self.infty_norm(K.permute(1,0)), self.infty_norm(V.permute(1,0))
        return self.maximum(u*v, v*u, u*w)
    def lipschitz_bound(self,x):
        B, C, width, height = x.size()
        N = width*height
        bound = torch.exp(torch.sqrt(torch.tensor(3, dtype=torch.float32))) * torch.sqrt(torch.tensor(N/C)) + torch.tensor(2, dtype=torch.float32)*torch.sqrt(torch.tensor(6, dtype=torch.float32))
        return bound.cuda()

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        if self.lip:
            energy /= self.norm_value.cuda()
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        if self.lip:
            out /= self.lipschitz_bound(out).cuda()
        out = self.gamma*out + x
        return out#,attention
class FCNet(nn.Module):

    def __init__(
        self, input_shape, idim, lipschitz_layer, nhidden, coeff, domains, codomains, n_iterations, activation_fn,
        preact, dropout, sn_atol, sn_rtol, learn_p, div_in=1
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        dim = c * h * w
        nnet = []
        last_dim = dim // div_in
        if preact:
            nnet.append(ACT_FNS[activation_fn](False))
        if learn_p:
            domains = [nn.Parameter(torch.tensor(0.))
                       for _ in range(len(domains))]
            codomains = domains[1:] + [domains[0]]
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                    last_dim, idim, coeff=coeff, n_iterations=n_iterations, domain=domains[
                        i], codomain=codomains[i],
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout:
            nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                last_dim, dim, coeff=coeff, n_iterations=n_iterations, domain=domains[-1], codomain=codomains[-1],
                atol=sn_atol, rtol=sn_rtol
            )
        )
        self.nnet = nn.Sequential(*nnet)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.nnet(x)
        return y.view(y.shape[0], *self.input_shape)


class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, logpx=None):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logpx is None:
            y = self.fc_module(x)
            return y.view(*shape)
        else:
            y, logpy = self.fc_module(x, logpx)
            return y.view(*shape), logpy

    def inverse(self, y, logpy=None):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logpy is None:
            x = self.fc_module.inverse(y)
            return x.view(*shape)
        else:
            x, logpx = self.fc_module.inverse(y, logpy)
            return x.view(*shape), logpx


class StackedCouplingBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        # yapf: disable
        class nonloc_scope:
            pass
        nonloc_scope.swap = True
        # yapf: enable

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _weight_layer(fc):
            return nn.Linear if fc else nn.Conv2d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                nonloc_scope.swap = not nonloc_scope.swap
                return layers.CouplingBlock(
                    initial_size[0],
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_weight_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        coeff=None,
                        domains=None,
                        codomains=None,
                        n_iterations=None,
                        sn_atol=None,
                        sn_rtol=None,
                        learn_p=None,
                        div_in=2,
                    ),
                    swap=nonloc_scope.swap,
                )
            else:
                ks = list(map(int, kernels.split('-')))

                if init_layer is None:
                    _block = layers.ChannelCouplingBlock
                    _mask_type = 'channel'
                    div_in = 2
                    mult_out = 1
                else:
                    _block = layers.MaskedCouplingBlock
                    _mask_type = 'checkerboard'
                    div_in = 1
                    mult_out = 2

                nonloc_scope.swap = not nonloc_scope.swap
                _mask_type += '1' if nonloc_scope.swap else '0'

                nnet = []
                if not first_resblock and preact:
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(_weight_layer(fc)(
                    initial_size[0] // div_in, idim, ks[0], 1, ks[0] // 2))
                if batchnorm:
                    nnet.append(layers.MovingBatchNorm2d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(_weight_layer(fc)(idim, idim, k, 1, k // 2))
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout:
                    nnet.append(nn.Dropout2d(dropout, inplace=True))
                nnet.append(_weight_layer(fc)(
                    idim, initial_size[0] * mult_out, ks[-1], 1, ks[-1] // 2))
                if batchnorm:
                    nnet.append(layers.MovingBatchNorm2d(initial_size[0]))

                return _block(initial_size[0], nn.Sequential(*nnet), mask_type=_mask_type)

        if init_layer is not None:
            chain.append(init_layer)
        if first_resblock and actnorm:
            chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm:
            chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, h, w = initial_size
            for i in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc,
                             first_resblock=first_resblock and (i == 0)))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm:
                        chain.append(_actnorm(initial_size, True))

        super(StackedCouplingBlocks, self).__init__(chain)