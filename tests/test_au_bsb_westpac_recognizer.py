import re

from predefined_recognizers.country_specific.australia.au_bsb_westpac_bank import (
    AuWestpacBsbRecognizer,
)


def test_westpac_bsb_regex_compiles():
    for pattern in AuWestpacBsbRecognizer.PATTERNS:
        re.compile(pattern.regex)


def test_westpac_bsb_validation():
    recognizer = AuWestpacBsbRecognizer()
    positives = [
        "03-1234",
        "031234",
        "03 1234",
        "731234",
        "73-1234",
        "73 1234",
    ]
    negatives = [
        "01-1234",
        "99 9999",
        "03123",
        "0312345",
        "ab-1234",
    ]

    for value in positives:
        assert recognizer.validate_result(value)

    for value in negatives:
        assert not recognizer.validate_result(value)
