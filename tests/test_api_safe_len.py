import types


from trend_analysis import api


class NoLen:
    pass


def test_safe_len_handles_sized_sequences():
    assert api._safe_len([1, 2, 3]) == 3
    assert api._safe_len("abc") == 3
    mapping = {"a": 1, "b": 2}
    assert api._safe_len(mapping) == 2


def test_safe_len_returns_zero_for_non_sized_objects():
    obj = NoLen()
    assert api._safe_len(obj) == 0

    class FakeSized:
        __len__ = None  # type: ignore[assignment]

    sized_like = types.SimpleNamespace(__len__=None)
    assert api._safe_len(sized_like) == 0
    assert api._safe_len(FakeSized()) == 0


def test_safe_len_respects_sized_protocol_without_len_attribute():
    class SizedProtocol:
        def __len__(self):
            return 7

    assert api._safe_len(SizedProtocol()) == 7
