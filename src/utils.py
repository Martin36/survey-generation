

def title_includes_search_strings(title: str):
    search_strings = ["survey", "systematic review", "literature review", "overview", ]
    # search_strings = ["survey"]
    match = False
    title = title.lower()

    for s in search_strings:
        if s in title:
            match = True

    return match


def sort_dict_keys_as_numbers(dictionary: dict):
    """Sorts a dict by key if the keys are number strings e.g. "1"

    Args:
        dictionary (dict): Dictionary to be sorted

    Returns:
        dict: The sorted dictionary
    """
    return dict(sorted(dictionary.items(), key=lambda t: int(t[0])))