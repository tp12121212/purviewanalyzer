from typing import List, Optional, Tuple

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer


class AuWestpacBsbRecognizer(PatternRecognizer):
    """
    Recognizes Westpac Bank-State-Branch ("BSB") number.

    In Australia, a BSB (Bank-State-Branch) is a 6 digit code which identifies a
    financial institution and branch, detected formats XXX-XXX, XX XXXX, XXXXXX, XXX XXX.

    Westpac BSBs commonly use a bank identifier prefix of "03" or "73" (e.g. 03x-xxx),
    with the remaining digits allocated to branches in line with AusPayNet BSB
    guidelines.

    This recognizer identifies Westpac BSB using regex and context words.
    Reference (official lookup): https://bsb.auspaynet.com.au/
    Reference (BSB system overview / downloads): https://auspaynet.com.au/BSBLinks

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    PATTERNS = [
        Pattern(
            "BSB Westpac (medium)",
            r"\b((0[3-4] [2-9]|0[3-4][2-9]|73 [0-9]|73[0-9])[ -]?\d{3,4})\b",
            0.1,
        )
    ]

    CONTEXT = [
        "Statement",
        "afsl",
        "Westpac",
        "233714",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "BSB_Westpac",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None,
        name: Optional[str] = None,
    ):
        self.replacement_pairs = (
            replacement_pairs if replacement_pairs else [("-", ""), (" ", "")]
        )
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name or "au_bsb_westpac_recognizer",
        )

    def validate_result(self, pattern_text: str) -> bool:
        """
        Validate the pattern logic e.g., by running checksum on a detected pattern.

        :param pattern_text: the text to validated.
        Only the part in text that was detected by the regex engine
        :return: A bool indicating whether the validation was successful.
        """
        # Pre-processing before validation checks
        text = EntityRecognizer.sanitize_value(pattern_text, self.replacement_pairs)
        if not text.isdigit() or len(text) != 6:
            return False
        return text.startswith(("03", "04","73"))
