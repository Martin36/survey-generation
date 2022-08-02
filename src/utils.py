from typing import List


def concat_input(input: List[List[str]], sep_token=None):
  """
  Concatenates the list of input documents for each input. 
  If sep_token is provided it will be added between the documents. 
  Otherwise, documents are only separated by a space.

  Args:
      input (List[str]): A list of input documents.
      sep_token (str, optional): The separation token. Defaults to None.

  Returns:
      _type_: _description_
  """
  result = []
  for d in input:
    input_str = ""
    for idx, doc in enumerate(d):
      if idx > 0:
        if sep_token:
          input_str += f" {sep_token} " + doc
        else:
          input_str += " " + doc
      else:
        input_str += doc
    result.append(input_str)
  return result


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