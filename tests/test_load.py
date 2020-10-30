import dp_mix.data.load as load


def test_load_faithful():
    assert load.load_faithful().shape == (272, 2)
