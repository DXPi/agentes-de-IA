from typing import Dict, List
import logging
import unittest
import json
from autogpt.memory_management_summary_memory import cleanup_new_events
from autogpt.config import Config

cfg = Config()
logger = logging.getLogger('test_log')
logger.setLevel(logging.ERROR)

class TestSummaryMemory(unittest.TestCase):
    def __init__(self):
        self.core_test_events = [
            {"role":"user", "content":"This event should disappear"},
            {"role":"system", "content":"This event's role should say 'your computer'."},
            {"role":"assistant", "content":"This will throw a JSON decode error"} ]
        self.bad_json_event = [ {
            "role":"assistant", 
            "content": "I suggest we start by browsing the repository to find any issues that we can fix. " + json.dumps( {
                "command": {
                    "name": "browse_website", 
                    "args":{ 
                        "url": "https://github.com/Torantulino/Auto-GPT" } },
                "thoughts":{
                    "text": "I suggest we start browsing the repository to find any issues that we can fix.",
                    "reasoning": "Browsing the repository will give us an idea of the current state of the codebase and identify any issues that we can address to improve the repo.",
                    "plan": "- Look through the repository to find any issues.\n- Investigate any issues to determine what needs to be fixed\n- Identify possible solutions to fix the issues\n- Open Pull Requests with fixes",
                    "criticism": "I should be careful while browsing so as not to accidentally introduce any new bugs or issues.",
                    "speak": "I will start browsing the repository to find any issues we can fix." } } ) } ]
        self.good_json_event = [ {
            "role":"assistant", 
            "content": json.dumps( {
                "command": {
                    "name": "browse_website",
                    "args": {"url": "https://github.com/Torantulino/Auto-GPT"},},
                "thoughts": {
                    "text": "I suggest we start browsing the repository to find any issues that we can fix.",
                    "reasoning": "Browsing the repository will give us an idea of the current state of the codebase and identify any issues that we can address to improve the repo.",
                    "plan": "- Look through the repository to find any issues.\n- Investigate any issues to determine what needs to be fixed\n- Identify possible solutions to fix the issues\n- Open Pull Requests with fixes",
                    "criticism": "I should be careful while browsing so as not to accidentally introduce any new bugs or issues.",
                    "speak": "I will start browsing the repository to find any issues we can fix." } } ) } ]

    def test_cleanup_new_events_fail(self):
        """Tests an event context that purposely fails to remove thoughts from an event, but successfully removes content:thoughts 
        from new_events and correctly modifies all other events."""
        fail_test = self.core_test_events + self.bad_json_event
        with self.assertLogs(level='ERROR') as logger:
            fail_new_events = cleanup_new_events(fail_test)
            assert logger.output[0].startswith("Error: Invalid JSON: ")
            assert '"thoughts":' in fail_new_events[2]['content']
            assert fail_new_events[0]['role'] == "your computer"
            assert fail_new_events[1]['role'] == "you"
            assert fail_new_events[2]['role'] == "you"


    def test_cleanup_new_events_pass(self):
        """Tests an event context that successfully removes content:thoughts and role:users from new_events, 
        and correctly modifies all other events."""
        pass_test =  self.core_test_events + self.good_json_event
        pass_new_events = cleanup_new_events(pass_test)
        assert '"thoughts":' not in pass_new_events[2]['content']
        assert pass_new_events[0]['role'] == "your computer"
        assert pass_new_events[1]['role'] == "you"
        assert pass_new_events[2]['role'] == "you"

    def test_no_new_events_pass(self):
        """Tests new_events when there are no new relevant events to add to the summary."""
        new_events =  [self.core_test_events[0]]
        new_events = cleanup_new_events(new_events)
        assert new_events == "Nothing new happened."
    
    def show_proof_that_latest_PR_still_fails(self):
        def old_code(new_events:List[Dict[str,str]]) -> List[Dict[str,str]]:
            """Shows proof that logic post- latest PR commit still does not function as expected"""
            # Replace "assistant" with "you". This produces much better first person past tense results.
            for event in new_events:
                if event["role"].lower() == "assistant":
                    event["role"] = "you"

                    # Remove "thoughts" dictionary from "content"
                    try:
                        content_dict = json.loads(event["content"])
                        if "thoughts" in content_dict:
                            del content_dict["thoughts"]
                        event["content"] = json.dumps(content_dict)
                    except json.decoder.JSONDecodeError:
                        if cfg.debug_mode:
                            logger.error(f"Error: Invalid JSON: {event['content']}\n")

                elif event["role"].lower() == "system":
                    event["role"] = "your computer"

                # Delete all user messages
                elif event["role"] == "user":
                    new_events.remove(event)

            # This can happen at any point during execution, not just the beginning
            if len(new_events) == 0:
                new_events = "Nothing new happened."
            return new_events
        pass_test =  self.core_test_events + self.good_json_event
        pass_new_events = old_code(pass_test)
        assert '"thoughts":' not in pass_new_events[2]['content']
        assert pass_new_events[0]['role'] == "your computer"
        assert pass_new_events[1]['role'] == "you"
        assert pass_new_events[2]['role'] == "you"
