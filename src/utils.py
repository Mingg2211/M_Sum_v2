from pycountry import languages

def normalize_language(language):
    for lookup_key in ("alpha_2", "alpha_3"):
        try:
            lang = languages.get(**{lookup_key: language})
            if lang:
                language = lang.name.lower()
        except KeyError:
            pass

    return language