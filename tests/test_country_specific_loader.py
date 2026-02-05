from presidio_analyzer import RecognizerRegistry

from app.recognizer_loader import load_country_specific_recognizers


def test_country_specific_recognizers_loaded():
    registry = RecognizerRegistry()
    load_country_specific_recognizers(registry)
    names = {rec.__class__.__name__ for rec in registry.recognizers}
    assert "AuWestpacBsbRecognizer" in names
