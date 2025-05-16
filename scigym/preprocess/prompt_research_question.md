# Systems Biology Research Challenge Generator

## Overview
You are tasked with creating meaningful research challenges to evaluate a scientist's ability to conduct biology research. Each challenge will present an incomplete SBML (Systems Biology Markup Language) model, representing partial understanding of a biological system. Participants must conduct wet lab experiments to discover the missing components.

Your task: Create three incomplete SBML models of varying difficulty by strategically masking components from a complete SBML model.

## Challenge Purpose
These challenges simulate authentic research scenarios where scientists must:
1. Formulate hypotheses about missing components
2. Design experiments to test these hypotheses
3. Systematically investigate through the scientific method

## SBML Masking Requirements

Masking means strategically removing specific components to create "gaps" in the model. When designing challenges:

* **Non-Trivial Complexity**: Create challenges that require multiple experiments and hypothesis iterations to solve, not simple pattern recognition.
* **Beyond Memorization**: The solution should not be a textbook pathway that could be recalled from memory but require active scientific reasoning.
* **Appropriate Scope**: A recommended guideline is to mask multiple 5-10 reactions or/and 3-5 species per challenge, though this is flexible based on the model complexity.

## Challenge Difficulty Levels

### LEVEL 1 (EASIEST): Parameter/Kinetic Law Estimation
- **Focus**: Quantitative aspects of known mechanisms
- **Challenge**: Determine reaction rates and enzyme kinetics
- **Available Function**: `remove_kinetic_law(reaction_id: str)`
- **Restriction**: You may NOT remove reactions or species at this level

### LEVEL 2 (INTERMEDIATE): Edge/Reaction Discovery
- **Focus**: Discover missing connections between known components
- **Challenge**: Infer reaction mechanisms between known species
- **Available Functions**:
  - `remove_reaction(reaction_id: str)`
- **Restriction**: You may NOT remove species at this level

### LEVEL 3 (CHALLENGING): Component/Species Discovery
- **Focus**: Discover entirely new components in the system
- **Challenge**: Infer the missing species and reactions
- **Available Functions**:
  - `remove_reaction(reaction_id: str)`
  - `remove_species(species_id: str)` - Removes a species and ALL reactions involving it


## Your Task

Using the provided SBML model, create THREE research challenges by strategically masking components:

1. Review the complete SBML model and its description provided below
2. Design ONE research challenge for EACH difficulty level
3. For each challenge, identify specific components to mask that would create an interesting scientific problem
4. Formulate research questions that focus on the resulting unexplained behavior
5. Specify the exact masking code required to implement each challenge


## Response Format

Your response must be a valid JSON object with the following structure:

```json
{
  "model_context": "A brief 1-3 sentence summary of the model and its biological context WITHOUT revealing the answers to the challenges",
  "biological_organisation": {
    "level": "The level of biological organisation you think this model belongs to. Your response has to be from this list:  Organelle, Cell, Tissue, Organ, Organism, Population, Ecosystem, Biosphere",
    "reasoning": "The reasons for your categorization."
  },
  "level_1_challenge": {
    "research_question": "An intriguing WHY or HOW question about unexplained biological phenomena",
    "detailed_description": "A thorough explanation of the biological context, what is being masked, why this creates an interesting research challenge, and what kinds of experiments might help solve it",
    "masking_code": [
      "remove_kinetic_law('reaction_id1')",
      "remove_kinetic_law('reaction_id2')",
      "remove_kinetic_law('reaction_id3')",
      "remove_kinetic_law('reaction_id4')",
      "remove_kinetic_law('reaction_id5')"
    ]
  },

  "level_2_challenge": {
    "research_question": "An intriguing WHY or HOW question about unexplained biological phenomena",
    "detailed_description": "A thorough explanation of the biological context, what is being masked, why this creates an interesting research challenge, and what kinds of experiments might help solve it",
    "masking_code": [
      "remove_reaction('reaction_id1')",
      "remove_reaction('reaction_id2')",
      "remove_reaction('reaction_id3')",
      "remove_reaction('reaction_id4')",
      "remove_reaction('reaction_id5')",
    ]
  },

  "level_3_challenge": {
    "research_question": "An intriguing WHY or HOW question about unexplained biological phenomena",
    "detailed_description": "A thorough explanation of the biological context, what is being masked, why this creates an interesting research challenge, and what kinds of experiments might help solve it",
    "masking_code": [
      "remove_reaction('reaction_id1')",
      "remove_reaction('reaction_id2')",
      "remove_species('species_id1')",
      "remove_species('species_id2')",
      "remove_species('species_id3')",
    ]
  }
}
```

## Critical Requirements

1. Research questions must reflect realistic challenges in systems biology
2. Your reaction_id and species_id MUST match the ones in the given sbml  
3. STRICTLY ADHERE to the function restrictions for each difficulty level
4. The "model_context" should provide sufficient background without revealing any challenge answers
5. You may remove multiple components for a single challenge (e.g., multiple kinetic laws, multiple reactions, or multiple species)
6. The `masking_code` field must be a list of function calls as strings, with each function call using the exact syntax shown in the difficulty levels section
7. The masking code must be ordered in a specific sequence: first list all `remove_kinetic_law` calls (if any), then all `remove_reaction` calls (if any), and finally all `remove_species` calls (if any) in this specific order.
