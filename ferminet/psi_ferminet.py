# Copyright 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid pair-biased attention networks for FermiNet."""

from typing import Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
from ferminet import network_blocks
from ferminet import networks
from ferminet import psiformer
import jax
import jax.numpy as jnp
import numpy as np


@attr.s(auto_attribs=True, kw_only=True)
class PsiFermiNetOptions(networks.BaseNetworkOptions):
    """Options controlling the Psi-FermiNet architecture."""

    num_layers: int = 2
    num_heads: int = 4
    heads_dim: int = 64
    mlp_hidden_dims: Tuple[int, ...] = (256,)
    pair_mlp_hidden_dims: Tuple[int, ...] = (64, 64)
    pair_embed_dim: int = 32
    use_layer_norm: bool = False
    use_pair_context: bool = False
    pair_spin_features: bool = True
    tf32: bool = False


def make_multi_head_attention(
    num_heads: int, heads_dim: int, tf32: bool = False
) -> ...:
    """Multi-head attention with optional additive pair bias."""
    prec = jax.lax.DotAlgorithmPreset.TF32_TF32_F32 if tf32 else None

    def linear_projection(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        y = jnp.dot(x, weights, precision=prec)
        return y.reshape(*x.shape[:-1], num_heads, heads_dim)

    def init(
        key: chex.PRNGKey, q_d: int, kv_d: int, output_channels: Optional[int] = None
    ) -> Mapping[str, jnp.ndarray]:
        qkv_hiddens = num_heads * heads_dim
        if not output_channels:
            output_channels = qkv_hiddens

        key, *subkeys = jax.random.split(key, num=4)
        params = {}
        params["q_w"] = network_blocks.init_linear_layer(
            subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]
        params["k_w"] = network_blocks.init_linear_layer(
            subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]
        params["v_w"] = network_blocks.init_linear_layer(
            subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
        )["w"]

        key, subkey = jax.random.split(key)
        params["attn_output"] = network_blocks.init_linear_layer(
            subkey, in_dim=qkv_hiddens, out_dim=output_channels, include_bias=False
        )["w"]

        return params

    def apply(
        params: networks.ParamTree,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        bias: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes attention with an optional additive logits bias."""
        q = linear_projection(query, params["q_w"])
        k = linear_projection(key, params["k_w"])
        v = linear_projection(value, params["v_w"])

        attn_logits = jnp.einsum("...thd,...Thd->...htT", q, k, precision=prec)
        attn_logits *= 1.0 / np.sqrt(heads_dim)
        if bias is not None:
            attn_logits = attn_logits + bias

        attn_weights = jax.nn.softmax(attn_logits)
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, v, precision=prec)
        attn = jnp.reshape(attn, (*query.shape[:-1], -1))
        return network_blocks.linear_layer(attn, params["attn_output"])

    return init, apply


def make_pair_mlp() -> ...:
    """Constructs an MLP for the two-electron stream."""

    def init(
        key: chex.PRNGKey,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
    ) -> Sequence[networks.Param]:
        params = []
        dims_in = [in_dim, *hidden_dims]
        dims_out = [*hidden_dims, out_dim]
        for dim_in, dim_out in zip(dims_in, dims_out):
            key, subkey = jax.random.split(key)
            params.append(
                network_blocks.init_linear_layer(
                    subkey, in_dim=dim_in, out_dim=dim_out, include_bias=True
                )
            )
        return params

    def apply(params: Sequence[networks.Param], inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, layer in enumerate(params):
            x = network_blocks.linear_layer(x, **layer)
            if i < len(params) - 1:
                x = jnp.tanh(x)
        return x

    return init, apply


def make_psi_ferminet_layers(
    nspins: Tuple[int, ...],
    natoms: int,
    options: PsiFermiNetOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
    """Creates the permutation-equivariant layers for Psi-FermiNet."""
    del nspins, natoms

    attn_dim = options.num_heads * options.heads_dim
    attention_init, attention_apply = make_multi_head_attention(
        options.num_heads, options.heads_dim, options.tf32
    )
    mlp_init, mlp_apply = psiformer.make_mlp()
    pair_mlp_init, pair_mlp_apply = make_pair_mlp()
    if options.use_layer_norm:
        layer_norm_init, layer_norm_apply = psiformer.make_layer_norm()

    def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
        params = {}
        feature_dims, params["input"] = options.feature_layer.init()
        one_electron_feature_dim, two_electron_feature_dim = feature_dims
        feature_dim = one_electron_feature_dim + 1
        pair_feature_dim = two_electron_feature_dim + int(options.pair_spin_features)

        key, subkey = jax.random.split(key)
        params["embed_one"] = network_blocks.init_linear_layer(
            subkey, in_dim=feature_dim, out_dim=attn_dim, include_bias=False
        )["w"]

        key, subkey = jax.random.split(key)
        params["embed_pair"] = network_blocks.init_linear_layer(
            subkey,
            in_dim=pair_feature_dim,
            out_dim=options.pair_embed_dim,
            include_bias=True,
        )

        attention_params = []
        mlp_params = []
        pair_mlp_params = []
        pair_bias_params = []
        ln_params = []
        pair_context_dim = 2 * attn_dim if options.use_pair_context else 0
        pair_input_dim = options.pair_embed_dim + pair_context_dim

        for _ in range(options.num_layers):
            key, attn_key, mlp_key, pair_key, bias_key = jax.random.split(key, 5)
            attention_params.append(
                attention_init(
                    attn_key, q_d=attn_dim, kv_d=attn_dim, output_channels=attn_dim
                )
            )
            mlp_params.append(mlp_init(mlp_key, options.mlp_hidden_dims, attn_dim))
            pair_mlp_params.append(
                pair_mlp_init(
                    pair_key,
                    in_dim=pair_input_dim,
                    hidden_dims=options.pair_mlp_hidden_dims,
                    out_dim=options.pair_embed_dim,
                )
            )
            pair_bias_params.append(
                network_blocks.init_linear_layer(
                    bias_key,
                    in_dim=options.pair_embed_dim,
                    out_dim=options.num_heads,
                    include_bias=True,
                )
            )
            if options.use_layer_norm:
                ln_params.append(
                    [layer_norm_init(attn_dim), layer_norm_init(attn_dim)]
                )

        params["attention"] = attention_params
        params["mlp"] = mlp_params
        params["pair_mlp"] = pair_mlp_params
        params["pair_bias"] = pair_bias_params
        params["ln"] = ln_params
        return attn_dim, params

    def apply(
        params,
        *,
        ae: jnp.ndarray,
        r_ae: jnp.ndarray,
        ee: jnp.ndarray,
        r_ee: jnp.ndarray,
        spins: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> jnp.ndarray:
        del charges

        ae_features, ee_features = options.feature_layer.apply(
            ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params["input"]
        )
        x = jnp.concatenate((ae_features, spins[..., None]), axis=-1)
        x = jnp.dot(x, params["embed_one"])

        y = ee_features
        if options.pair_spin_features:
            pair_spin = jnp.equal(spins[:, None], spins[None, :]).astype(y.dtype)
            y = jnp.concatenate((y, pair_spin[..., None]), axis=-1)
        y = network_blocks.linear_layer(y, **params["embed_pair"])

        for layer in range(options.num_layers):
            if options.use_pair_context:
                xi = jnp.broadcast_to(x[:, None, :], (x.shape[0], x.shape[0], x.shape[1]))
                xj = jnp.broadcast_to(x[None, :, :], (x.shape[0], x.shape[0], x.shape[1]))
                y_in = jnp.concatenate((y, xi, xj), axis=-1)
            else:
                y_in = y

            y = pair_mlp_apply(params["pair_mlp"][layer], y_in)
            pair_bias = network_blocks.linear_layer(y, **params["pair_bias"][layer])
            pair_bias = jnp.transpose(pair_bias, (2, 0, 1))

            attn_output = attention_apply(
                params["attention"][layer], x, x, x, bias=pair_bias
            )
            x = x + attn_output
            if options.use_layer_norm:
                x = layer_norm_apply(params["ln"][layer][0], x)

            mlp_output = mlp_apply(params["mlp"][layer], x)
            x = x + mlp_output
            if options.use_layer_norm:
                x = layer_norm_apply(params["ln"][layer][1], x)

        return x

    return init, apply


def make_fermi_net(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    num_layers: int = 4,
    num_heads: int = 4,
    heads_dim: int = 64,
    mlp_hidden_dims: Tuple[int, ...] = (256,),
    pair_mlp_hidden_dims: Tuple[int, ...] = (64, 64),
    pair_embed_dim: int = 32,
    use_layer_norm: bool = True,
    use_pair_context: bool = False,
    pair_spin_features: bool = True,
    tf32: bool = False,
) -> networks.Network:
    """Psi-FermiNet with pair-biased self-attention layers."""
    if not envelope:
        envelope = envelopes.make_isotropic_envelope()

    if not feature_layer:
        natoms = charges.shape[0]
        feature_layer = networks.make_ferminet_features(
            natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
        )

    if isinstance(jastrow, str):
        if jastrow.upper() == "DEFAULT":
            jastrow = jastrows.JastrowType.SIMPLE_EE
        else:
            jastrow = jastrows.JastrowType[jastrow.upper()]

    options = PsiFermiNetOptions(
        ndim=ndim,
        determinants=determinants,
        states=states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=jastrow,
        complex_output=complex_output,
        bias_orbitals=bias_orbitals,
        full_det=True,
        rescale_inputs=rescale_inputs,
        num_layers=num_layers,
        num_heads=num_heads,
        heads_dim=heads_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        pair_mlp_hidden_dims=pair_mlp_hidden_dims,
        pair_embed_dim=pair_embed_dim,
        use_layer_norm=use_layer_norm,
        use_pair_context=use_pair_context,
        pair_spin_features=pair_spin_features,
        tf32=tf32,
    )

    equivariant_layers = make_psi_ferminet_layers(nspins, charges.shape[0], options)
    orbitals_init, orbitals_apply = networks.make_orbitals(
        nspins=nspins,
        charges=charges,
        options=options,
        equivariant_layers=equivariant_layers,
    )

    def network_init(key: chex.PRNGKey) -> networks.ParamTree:
        return orbitals_init(key)

    def network_apply(
        params,
        pos: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        orbitals = orbitals_apply(params, pos, spins, atoms, charges)
        if options.states:
            batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
            orbitals = [
                jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
                for orbital in orbitals
            ]
            result = batch_logdet_matmul(orbitals)
        else:
            result = network_blocks.logdet_matmul(orbitals)
        if "state_scale" in params:
            result = result[0], result[1] + params["state_scale"]
        return result

    return networks.Network(
        options=options,
        init=network_init,
        apply=network_apply,
        orbitals=orbitals_apply,
    )
