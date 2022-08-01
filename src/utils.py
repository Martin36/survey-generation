
def match_search_terms(text: str):
	"""Looks for specific search terms in a string
	
	Args:
		text (str): String to be searched

	Returns:
		bool: True if the string contains any of the search terms
	"""
	if not text:
		return False
	text = text.lower()
	search_terms = ["survey", "review", "overview", "shared task"]
	for term in search_terms:
		if term in text:
			return True
	return False


def sort_dict_keys_as_numbers(dictionary: dict):
	"""Sorts a dict by key if the keys are number strings e.g. "1"

	Args:
		dictionary (dict): Dictionary to be sorted

	Returns:
		dict: The sorted dictionary
	"""
	return dict(sorted(dictionary.items(), key=lambda t: int(t[0])))