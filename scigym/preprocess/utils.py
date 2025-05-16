import os
import re
from typing import Dict

import libsbml
import requests

from scigym.api import (
    CreateQuestionAction,
    RemoveKineticLawAction,
    RemoveReactionAction,
    RemoveSpeciesAction,
)
from scigym.constants import MODEL_TO_API_KEY_NAME
from scigym.llm import LLM


def query_llm(
    prompt: str, model_name: str = "gemini-2.5-pro-preview-03-25", system_prompt: str = ""
):
    api_key = os.environ.get(MODEL_TO_API_KEY_NAME[model_name])
    if api_key is None:
        raise ValueError(f"Did not find an API key for model: {model_name}")
    llm = LLM(model_name, api_key, system_prompt)
    llm.initialize_chat()
    return llm.return_response(prompt)


def query_claude(prompt, sbml_path=None):
    """
    Query Claude API with the given prompt.
    If sbml_path is provided, include the SBML model in the prompt.

    Args:
        prompt (str): The prompt to send to Claude
        sbml_path (str, optional): Path to the SBML model file

    Returns:
        str: Claude's response
    """
    # If sbml_path is provided, read the SBML model and include it in the prompt
    if sbml_path:
        with open(sbml_path, "r") as file:
            sbml_content = file.read()

        # Augment the prompt with the SBML model
        prompt = prompt.replace("{SBML_MODEL}", sbml_content)

    # Here you would implement the actual API call to Claude
    # This is a placeholder for the actual implementation
    api_key = os.environ.get("CLAUDE_API_KEY")
    api_url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    data = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["content"][0]["text"]
    else:
        raise Exception(f"Error querying Claude: {response.text}")


def parse_action_string(
    action_string: str, id_mapping_dict: Dict[str, Dict[str, str]]
) -> CreateQuestionAction:
    """
    Parse a string representation of an action into the appropriate dataclass object.

    Supported formats:
    - remove_kinetic_law('reaction_id1')    # Direct parameter with single quotes
    - remove_reaction("reaction_id2")       # Direct parameter with double quotes
    - remove_species('species_id3')

    Args:
        action_string: A string representation of the action
        id_mapping_dict: A dictionary mapping real IDs to fake IDs by sbase type code

    Returns:
        An instance of RemoveReactionAction, RemoveSpeciesAction, or RemoveKineticLawAction

    Raises:
        ValueError: If the string format is not recognized
    """
    # Extract the action type (function name)
    action_match = re.match(r"(\w+)\(", action_string.strip())
    if not action_match:
        raise ValueError(f"Invalid action string format: {action_string}")

    action_type = action_match.group(1)

    # Extract the parameter (supporting both single and double quotes)
    # Using a regex pattern that matches either 'value' or "value"
    param_pattern = r'\w+\(([\'"])([^\'"]+)\1\)'
    param_match = re.match(param_pattern, action_string.strip())

    if not param_match:
        raise ValueError(f"Invalid parameter format in action string: {action_string}")

    # The second group contains the actual parameter value (without quotes)
    param_value = param_match.group(2)

    reaction_mapping = id_mapping_dict.get(str(libsbml.SBML_REACTION), {})
    species_mapping = id_mapping_dict.get(str(libsbml.SBML_SPECIES), {})

    # Create the appropriate dataclass object based on the action type
    if action_type == "remove_reaction":
        param_value = reaction_mapping.get(param_value, param_value)
        return RemoveReactionAction(reaction_id=param_value)
    elif action_type == "remove_species":
        param_value = species_mapping.get(param_value, param_value)
        return RemoveSpeciesAction(species_id=param_value)
    elif action_type == "remove_kinetic_law":
        param_value = reaction_mapping.get(param_value, param_value)
        return RemoveKineticLawAction(reaction_id=param_value)
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def merge_two_nested_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """Merge two nested dictionaries by updating dict1 with dict2"""
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, dict):
                dict1[key] = merge_two_nested_dictionaries(dict1[key], value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1
