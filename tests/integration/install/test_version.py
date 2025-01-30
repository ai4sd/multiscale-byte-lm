import re

import mblm


def test_version():
    version_re_pattern = re.compile(r"\d\.\d\.\d")
    assert version_re_pattern.match(mblm.__version__)
