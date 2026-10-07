# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""einsums:packed-gemm:batch-promotion changes the route, never the answer.

"mcs <- mbs ; bc" has a target index (s) carried by A alone, so its M group is
(m, s). The packed engine runs it through its scatter kernels; promoted, it is
one gemm_batch, a (m x b)(b x c) GEMM per s against the same B. The default
mode promotes only thin GEMMs (b = c <= 128 threaded, <= 32 on one thread), so
16 and 160 take different routes under it, and modes -1 and 1 force each route
for both sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.rc as rc


@pytest.fixture
def restore_promotion():
    saved = rc.packed_gemm_batch_promotion
    yield
    rc.packed_gemm_batch_promotion = saved


def test_batch_promotion_default_is_auto():
    assert rc.packed_gemm_batch_promotion in (None, 0)


@pytest.mark.parametrize("mode", [-1, 0, 1])
@pytest.mark.parametrize("nb", [16, 160])
def test_every_mode_matches_numpy(restore_promotion, mode, nb):
    M, S = 96, 6
    rc.packed_gemm_batch_promotion = mode
    X = einsums.create_random_tensor("X", [M, nb * S])
    U = einsums.create_random_tensor("U", [nb, nb])
    Y = einsums.create_zero_tensor("Y", [M, nb * S])
    X3, Y3 = X.reshape_view([M, nb, S]), Y.reshape_view([M, nb, S])

    einsums.einsum("mcs <- mbs ; bc", Y3, X3, U, c_pf=0.0, ab_pf=1.0)

    # einsums is column-major: the flat index of (m, b, s) is m + M*(b + nb*s).
    x = np.asarray(X).reshape(M, nb * S, order="F").reshape(M, nb, S, order="F")
    y = np.asarray(Y).reshape(M, nb * S, order="F").reshape(M, nb, S, order="F")
    np.testing.assert_allclose(y, np.einsum("mbs,bc->mcs", x, np.asarray(U)), rtol=1e-12, atol=1e-12)
