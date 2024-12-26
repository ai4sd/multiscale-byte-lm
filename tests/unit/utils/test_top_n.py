from mblm.utils.top_n import TopN


class TestTopN:
    def test_top_n_docstring_example(self):
        # create the heap
        top_n = TopN[str](2)

        # add items in random order
        top_n.add((1, "a"))
        top_n.add((3, "b"))
        top_n.add((2, "c"))

        for idx, item in enumerate(top_n):
            ...
            # iteration 1: (1, "a")
            # iteration 2: (2, "c")

        it = iter(top_n)
        assert next(it) == (1, "a")
        assert next(it) == (2, "c")
        assert top_n.get_top(1) == [(1, "a")]

    def test_top_n_min(self):
        num_items = 2
        top_n = TopN[str](num_items)
        top_n.add((1.0, "1"))
        top_n.add((3.0, "3"))
        top_n.add((2.0, "2"))
        assert len(top_n.get_top()) == num_items
        assert len(top_n.get_top()) == len(top_n)
        assert top_n.get_top(1) == [(1.0, "1")]
        assert top_n.get_top(2) == [(1.0, "1"), (2.0, "2")]

        it = iter(top_n)
        assert next(it) == (1.0, "1")
        assert next(it) == (2.0, "2")

        # add more items, which should go in front
        top_n.add((0, "0"))
        assert len(top_n.get_top()) == num_items
        it = iter(top_n)
        assert next(it) == (0.0, "0")
        assert next(it) == (1.0, "1")

    def test_top_n_max(self):
        num_items = 2
        top_n = TopN[str](num_items, top_largest=True)
        top_n.add((1.0, "1"))
        top_n.add((3.0, "3"))
        top_n.add((2.0, "2"))
        assert len(top_n.get_top()) == num_items
        assert len(top_n.get_top()) == len(top_n)
        assert top_n.get_top(1) == [(3.0, "3")]
        assert top_n.get_top(2) == [(3.0, "3"), (2.0, "2")]

        it = iter(top_n)
        assert next(it) == (3.0, "3")
        assert next(it) == (2.0, "2")

        # add more items, which should go to the back
        top_n.add((4, "4"))
        assert len(top_n.get_top()) == num_items
        it = iter(top_n)
        assert next(it) == (4.0, "4")
        assert next(it) == (3.0, "3")

    def test_top_n_min_equal_stores_first(self):
        num_items = 3
        top_n = TopN[str](num_items)
        top_n.add((2.0, "2a"))
        top_n.add((3.0, "3"))
        top_n.add((1.0, "1"))
        top_n.add((2.0, "2b"))
        assert len(top_n.get_top()) == num_items

        it = iter(top_n)
        assert next(it) == (1.0, "1")
        assert next(it) == (2.0, "2a")
        assert next(it) == (2.0, "2b")

    def test_top_n_min_deep_copy(self):
        top_n_shallow = TopN[list[int]](1)
        lst = [8]
        top_n_shallow.add((1.0, lst))

        # mutate a reference in place
        lst[0] = 9
        assert top_n_shallow.get_top(1) == [(1.0, [9])]
        assert top_n_shallow.get_top(1)[0][1] is lst

        top_n_deep = TopN[list[int]](1, deep_copy=True)
        top_n_deep.add((1.0, lst))

        # mutate a reference in place back to original
        lst[0] = 8
        assert top_n_deep.get_top(1) == [(1.0, [9])]
        assert top_n_deep.get_top(1)[0][1] is not lst
