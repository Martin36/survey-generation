import unittest, jsonlines

from utils_package.util_funcs import load_json, load_jsonl

from explore_s2orc import filter_acl_survey_papers, filter_computer_science_survey_papers, title_includes_search_strings


class TestExploreS2ORC(unittest.TestCase):

    def test_title_includes_search_strings(self):
        title = "A survey of some random topic"
        self.assertTrue(title_includes_search_strings(title), "Should be True when containing the string 'survey'")

        title = "Just a regular paper"
        self.assertFalse(title_includes_search_strings(title), "Should be False when not mentioning 'survey' in the title")

        title = "A Survey of Some Random Topic"
        self.assertTrue(title_includes_search_strings(title), "Should NOT be case sensitive")


    def test_filter_acl_survey_papers(self):
        data_file = "data/metadata/sample.jsonl"
        data = load_jsonl(data_file)
        
        acl_papers = [d for d in data if d["acl_id"] is not None]
        arxiv_papers = [d for d in data if d["arxiv_id"] is not None]
        survey_papers = [d for d in data if "survey" in d["title"].lower()]
        # TODO: There is no ACL papers in the sample data, so it is difficult to test this
        is_survey_paper = filter_acl_survey_papers(data[0])

    
    def test_filter_computer_science_survey_papers(self):
        acl_paper = load_json("test_data/acl_paper.json")
        medicine_paper = load_json("test_data/medicine_paper.json")
        paper_without_category = acl_paper.copy()
        paper_without_category["mag_field_of_study"] = None

        self.assertTrue(filter_computer_science_survey_papers(acl_paper), "Should be True when paper is in the Computer Science domain")
        self.assertFalse(filter_computer_science_survey_papers(medicine_paper), "Should be False when paper is in a non Computer Science domain")
        self.assertFalse(filter_computer_science_survey_papers(paper_without_category), "Should be False when paper doesn't have a category")

    

        
if __name__ == '__main__':
    unittest.main()