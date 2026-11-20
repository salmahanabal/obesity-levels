from obesity.sampling import INT_RANGES, random_samples


def test_random_samples_valid_and_reproducible():
    a, b = random_samples(50, seed=1), random_samples(50, seed=1)
    assert a.equals(b)
    assert a["Age"].between(14, 65).all() and a["Height"].between(1.3, 2.1).all()
    for k, (lo, hi) in INT_RANGES.items():
        assert a[k].between(lo, hi).all()
