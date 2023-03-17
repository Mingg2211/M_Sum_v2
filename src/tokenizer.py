# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import re
import string
import zipfile

import nltk

from _compat import to_string, to_unicode, unicode
from utils import normalize_language

class DefaultWordTokenizer(object):
    """NLTK tokenizer"""
    def tokenize(self, text):
        return nltk.word_tokenize(text)


class ChineseWordTokenizer:
    def tokenize(self, text):
        try:
            import jieba
        except ImportError as e:
            raise ValueError("Chinese tokenizer requires jieba. Please, install it by command 'pip install jieba'.")
        return jieba.cut(text)

class VietNamWordTokenizer:
    def tokenize(self, text):
        try:
            from underthesea import word_tokenize
        except ImportError as e:
            raise ValueError("VietNam tokenizer requires underthesea. Please, install it by command 'pip install underthesea'.")
        return word_tokenize(text)
    
class RussianWordTokenizer:
    def tokenize(self, text):
        try:
            import nltk
        except ImportError as e:
            raise ValueError("Russian tokenizer requires nltk. Please, install it by command 'pip install nltk'.")
        return nltk.word_tokenize(text, language='russian')
class VietNameseSentencesTokenizer:
    def tokenize(self, text):
        try:
            from underthesea import sent_tokenize
        except ImportError as e:
            raise ValueError("VietNamese tokenizer requires underthesea. Please, install it by command 'pip install underthesea'.")
        return sent_tokenize(text)

class Tokenizer(object):
    """Language dependent tokenizer of text document."""

    _WORD_PATTERN = re.compile(r"^[^\W\d_](?:[^\W\d_]|['-])*$", re.UNICODE)
    # feel free to contribute if you have better tokenizer for any of these languages :)
    LANGUAGE_ALIASES = {
        "slovak": "czech",
    }

    # improve tokenizer by adding specific abbreviations it has issues with
    # note the final point in these items must not be included
    LANGUAGE_EXTRA_ABREVS = {
        "english": ["e.g", "al", "i.e"],
        "russian": ["ім.", "о.", "вул.", "просп.", "бул.", "пров.", "пл.", "г.", "р.", "див.", "п.", "с.", "м."],
        "vietnamese": ['v.v'],
    }

    SPECIAL_SENTENCE_TOKENIZERS = {
        'russian': nltk.RegexpTokenizer(r'[.!?…»]', gaps=True),
        'chinese': nltk.RegexpTokenizer('[^　！？。]*[！？。]'),
        'vietnamese' : VietNameseSentencesTokenizer(),
    }

    SPECIAL_WORD_TOKENIZERS = {

        'chinese': ChineseWordTokenizer(),
        'vietnamese': VietNamWordTokenizer(),
        'russian':RussianWordTokenizer(),
    }

    def __init__(self, language):
        language = normalize_language(language)
        self._language = language

        tokenizer_language = self.LANGUAGE_ALIASES.get(language, language)
        self._sentence_tokenizer = self._get_sentence_tokenizer(tokenizer_language)
        self._word_tokenizer = self._get_word_tokenizer(tokenizer_language)

    @property
    def language(self):
        return self._language

    def _get_sentence_tokenizer(self, language):
        if language in self.SPECIAL_SENTENCE_TOKENIZERS:
            return self.SPECIAL_SENTENCE_TOKENIZERS[language]
        try:
            path = to_string("tokenizers/punkt/%s.pickle") % to_string(language)
            return nltk.data.load(path)
        except (LookupError, zipfile.BadZipfile) as e:
            raise LookupError(
                "NLTK tokenizers are missing or the language is not supported.\n"
                """Download them by following command: python -c "import nltk; nltk.download('punkt')"\n"""
                "Original error was:\n" + str(e)
            )

    def _get_word_tokenizer(self, language):
        if language in self.SPECIAL_WORD_TOKENIZERS:
            return self.SPECIAL_WORD_TOKENIZERS[language]
        else:
            return DefaultWordTokenizer()

    def to_sentences(self, paragraph):
        if hasattr(self._sentence_tokenizer, '_params'):
            extra_abbreviations = self.LANGUAGE_EXTRA_ABREVS.get(self._language, [])
            self._sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
        sentences = self._sentence_tokenizer.tokenize(to_unicode(paragraph))
        return tuple(map(unicode.strip, sentences))

    def to_words(self, sentence):
        words = self._word_tokenizer.tokenize(to_unicode(sentence))
        return tuple(filter(self._is_word, words))

    @staticmethod
    def _is_word(word):
        return bool(Tokenizer._WORD_PATTERN.match(word))
