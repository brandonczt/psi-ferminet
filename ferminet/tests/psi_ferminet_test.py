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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import psi_ferminet
import jax
import jax.numpy as jnp
import numpy as np


def _network_options():
    all_options = {
        "vmap": [True, False],
        "ndim": [2, 3],
        "determinants": [1, 4],
        "nspins": [(1, 1), (1, 0), (2, 1)],
        "jastrow": ["none", "simple_ee"],
        "rescale_inputs": [True, False],
    }
    for options in itertools.product(*all_options.values()):
        yield dict(zip(all_options.keys(), options))


class PsiFermiNetTest(parameterized.TestCase):
    @parameterized.parameters(
        {"jastrow": "simple_ee"},
        {"jastrow": "none"},
    )
    def test_antisymmetry(self, jastrow):
        key = jax.random.PRNGKey(42)
        natom = 4
        key, *subkeys = jax.random.split(key, num=3)
        atoms = jax.random.normal(subkeys[0], shape=(natom, 3))
        charges = jax.random.normal(subkeys[1], shape=(natom,))
        nspins = (3, 4)
        determinants = 6

        network = psi_ferminet.make_fermi_net(
            nspins,
            charges,
            determinants=determinants,
            ndim=3,
            jastrow=jastrow,
            num_layers=3,
            num_heads=4,
            heads_dim=32,
            mlp_hidden_dims=(64, 128),
            pair_mlp_hidden_dims=(32, 32),
            pair_embed_dim=16,
            use_layer_norm=True,
            use_pair_context=True,
            pair_spin_features=True,
            tf32=False,
        )

        key, subkey = jax.random.split(key)
        params = network.init(subkey)

        for i in range(len(params["envelope"])):
            if params["envelope"][i]:
                key, *subkeys = jax.random.split(key, num=3)
                params["envelope"][i]["sigma"] = jax.random.normal(
                    subkeys[0], params["envelope"][i]["sigma"].shape
                )
                params["envelope"][i]["pi"] = jax.random.normal(
                    subkeys[1], params["envelope"][i]["pi"].shape
                )

        key, subkey = jax.random.split(key)
        pos1 = jax.random.normal(subkey, shape=(sum(nspins) * 3,))
        pos2 = jnp.concatenate((pos1[3:6], pos1[:3], pos1[6:]))
        pos3 = jnp.concatenate((pos1[:9], pos1[12:15], pos1[9:12], pos1[15:]))

        key, subkey = jax.random.split(key)
        spins1 = jax.random.uniform(subkey, shape=(sum(nspins),))
        spins2 = jnp.concatenate((spins1[1:2], spins1[:1], spins1[2:]))
        spins3 = jnp.concatenate((spins1[:3], spins1[4:5], spins1[3:4], spins1[5:]))

        out1 = network.apply(params, pos1, spins1, atoms, charges)

        if out1[0].dtype == jnp.float32:
            rtol = 1.5e-4
            atol = 1.0e-4
        else:
            rtol = 1.0e-7
            atol = 0

        out2 = network.apply(params, pos2, spins2, atoms, charges)
        np.testing.assert_allclose(out1[1], out2[1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(out1[0], -1 * out2[0])

        out3 = network.apply(params, pos3, spins3, atoms, charges)
        np.testing.assert_allclose(out1[1], out3[1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(out1[0], -1 * out3[0])

    @parameterized.parameters(_network_options())
    def test_psi_ferminet(self, **network_options):
        nspins = network_options["nspins"]
        ndim = network_options["ndim"]
        atoms_3d = jnp.asarray([[0.0, 0.0, 0.2], [1.2, 1.0, -0.2], [2.5, -0.8, 0.6]])
        atoms = atoms_3d[:, :ndim]
        charges = jnp.asarray([2, 5, 7])
        key = jax.random.PRNGKey(42)

        network = psi_ferminet.make_fermi_net(
            nspins,
            charges,
            determinants=network_options["determinants"],
            ndim=ndim,
            rescale_inputs=network_options["rescale_inputs"],
            jastrow=network_options["jastrow"],
            num_layers=3,
            num_heads=4,
            heads_dim=32,
            mlp_hidden_dims=(64, 32),
            pair_mlp_hidden_dims=(32,),
            pair_embed_dim=16,
            use_layer_norm=True,
            use_pair_context=True,
            pair_spin_features=True,
            tf32=False,
        )

        key, subkey = jax.random.split(key)
        if network_options["vmap"]:
            batch = 10
            xs = jax.random.uniform(subkey, shape=(batch, sum(nspins), ndim))
            network_apply = jax.vmap(network.apply, in_axes=(None, 0, 0, None, None))
            expected_shape = (batch,)
        else:
            batch = 1
            xs = jax.random.uniform(subkey, shape=(sum(nspins), ndim))
            network_apply = network.apply
            expected_shape = ()

        key, subkey = jax.random.split(key)
        spins = jax.random.uniform(subkey, shape=(batch, sum(nspins)))
        if not network_options["vmap"]:
            spins = jnp.squeeze(spins, axis=0)

        key, subkey = jax.random.split(key)
        params = network.init(subkey)

        sign_out, log_out = network_apply(params, xs, spins, atoms, charges)
        self.assertSequenceEqual(sign_out.shape, expected_shape)
        self.assertSequenceEqual(log_out.shape, expected_shape)


if __name__ == "__main__":
    absltest.main()
