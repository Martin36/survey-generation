

def title_includes_search_strings(title: str):
    # search_strings = ["survey", "systematic review", "literature review"]
    search_strings = ["survey"]
    match = False
    title = title.lower()

    for s in search_strings:
        if s in title:
            match = True

    return match
