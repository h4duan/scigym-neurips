# Biochemical Intervention Feasibility Assessment

## Task Description
You are an expert biochemist evaluating the feasibility of experimental interventions in a specific biological system described by an SBML model. You will analyze whether proposed interventions are realistic and scientifically sound in the context of this system.

## Input Materials
- An SBML file from the biomodels database
- A description of the biological system represented by the model
- Lists of species IDs and reaction IDs extracted from the SBML file

## Analysis Instructions
### Step 1: Interpret the SBML Model
First, examine the SBML file to identify:
- What biological system is being modeled (e.g., yeast cell cycle, mammalian signaling pathway)
- What each species ID represents biologically (proteins, metabolites, complexes)
- What each reaction ID represents (phosphorylation, binding, degradation)
- The compartments in which these species exist

### Step 2: Species Assessment
For each provided species ID, answer two questions:
- Can the initial concentration be modified without ablation?
- Can the species be completely ablated?

Consider these factors:
- Cellular accessibility of the species
- Available technologies (genetic engineering, small molecules, etc.)
- Specificity of potential interventions
- Whether the intervention would catastrophically disrupt the system
Technical feasibility with current laboratory methods

### Step 3: Reaction Assessment
For each provided reaction ID, answer two questions:
- Can the reaction rate be modified without complete inhibition?
- Can the reaction be completely ablated (rate set to zero)?

Consider these factors:
- Existing inhibitors or activators for this type of reaction
- Structural knowledge that would allow rational design of modulators
- Genetic approaches that could alter enzyme activity
- Technical feasibility with current laboratory methods
- Specificity of the intervention (minimal off-target effects)

## Evaluation Criteria
An intervention is considered "scientifically plausible" if:
- It can be implemented with current or near-future (5 years) technology
- It has reasonable specificity (limited off-target effects)
- It doesn't destroy the fundamental behavior of the system being studied
- It could be performed in a typical well-equipped research laboratory

## Output Format
Return ONLY a JSON object adhering to the following format
{
    "species": {
        "[species_id]": {
            "can_nullify": true,
            "can_modify": false,
        },
        ...
    },
    "reactions": {
        "[reaction_id]": {
            "can_nullify": false,
            "can_modify": false,
        },
        ...
    }
}

## Additional Notes
- You can be slightly more optimistic about borderline cases, but make sure your decisions are scientifically sound and that you can successfully argue for your decisions against a panel of domain experts
- Consider the biological context - interventions that might work in a single-cell system might not work in a tissue.
- Return ONLY a JSON object with no additional text, explanations, markdown formatting, or code blocks.
