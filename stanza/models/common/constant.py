"""
Global constants.
"""

lcode2lang = {
    "af": "Afrikaans",
    "grc": "Ancient_Greek",
    "ar": "Arabic",
    "hy": "Armenian",
    "eu": "Basque",
    "be": "Belarusian",
    "br": "Breton",
    "bg": "Bulgarian",
    "bxr": "Buryat",
    "ca": "Catalan",
    "zh-hant": "Traditional_Chinese",
    "lzh": "Classical_Chinese",
    "cop": "Coptic",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fo": "Faroese",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "got": "Gothic",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "qhe": "Hindi_English",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "ko": "Korean",
    "kmr": "Kurmanji",
    "lt": "Lithuanian",
    "olo": "Livvi",
    "la": "Latin",
    "lv": "Latvian",
    "mt": "Maltese",
    "mr": "Marathi",
    "pcm": "Naija",
    "sme": "North_Sami",
    # there used to be different mappings for different norwegian treebanks,
    # but that seems to be updated as of 2.7
    "no": "Norwegian",
    "cu": "Old_Church_Slavonic",
    "fro": "Old_French",
    "orv": "Old_Russian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "gd": "Scottish_Gaelic",
    "sr": "Serbian",
    "zh-hans": "Simplified_Chinese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "swl": "Swedish_Sign_Language",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "qtd": "Turkish_German",
    "uk": "Ukrainian",
    "hsb": "Upper_Sorbian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "vi": "Vietnamese",
    "wo": "Wolof"
}

lang2lcode = {lcode2lang[k]: k for k in lcode2lang}
langlower2lcode = {lcode2lang[k].lower(): k.lower() for k in lcode2lang}

# additional useful code to language mapping
# added after dict invert to avoid conflict
# this mapping seems obsolete?  it all maps to "no" in 2.7
# lcode2lang['nb'] = 'Norwegian' # Norwegian Bokmall mapped to default norwegian
lcode2lang['zh'] = 'Simplified_Chinese'

lang2lcode['Chinese'] = 'zh'

def treebank_to_short_name(treebank):
    """ Convert treebank name to short code. """
    if treebank.startswith('UD_'):
        treebank = treebank[3:]
    splits = treebank.split('-')
    assert len(splits) == 2
    lang, corpus = splits

    lcode = lang2lcode[lang]

    short = "{}_{}".format(lcode, corpus.lower())
    return short

