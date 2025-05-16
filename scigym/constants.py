import libsbml

# Type-specific configuration for metadata removal
DEFAULT_METADATA_REMOVAL_CONFIG = {
    # Default configuration for all SBase objects
    "default": {
        "del_name": False,
        "del_metaid": True,
        "del_notes": True,
        "del_annotations": True,
        "del_history": True,
        "del_sbo_terms": True,
        "del_cv_terms": True,
        "del_created_date": True,
        "del_modified_date": True,
        "del_user_data": True,
    },
    # Override specific settings for Species
    libsbml.SBML_SPECIES: {
        "del_annotations": False,
        "del_cv_terms": False,
        "del_sbo_terms": False,
    },
    # Override specific settings for Compartment
    libsbml.SBML_COMPARTMENT: {
        "del_annotations": False,
        "del_cv_terms": False,
        "del_sbo_terms": False,
    },
    # Add more type-specific overrides as needed
    # ExampleClass: {
    #     "del_something": False,
    # },
}


ANONYMIZE_EVERYTHING = {
    "default": {
        "del_name": True,
        "del_metaid": True,
        "del_notes": True,
        "del_annotations": True,
        "del_history": True,
        "del_sbo_terms": True,
        "del_cv_terms": True,
        "del_created_date": True,
        "del_modified_date": True,
        "del_user_data": True,
    },
}


ANONYMIZE_EVERYTHING_EXCEPT_SPECIES_CONFIG = {
    "default": {
        "del_name": True,
        "del_metaid": True,
        "del_notes": True,
        "del_annotations": True,
        "del_history": True,
        "del_sbo_terms": True,
        "del_cv_terms": True,
        "del_created_date": True,
        "del_modified_date": True,
        "del_user_data": True,
    },
    libsbml.SBML_SPECIES: {
        "del_name": False,
    },
}


ANONYMIZE_METADATA_REMOVAL_CONFIG = {
    "default": {
        "del_name": True,
        "del_metaid": True,
        "del_notes": True,
        "del_annotations": True,
        "del_history": True,
        "del_sbo_terms": True,
        "del_cv_terms": True,
        "del_created_date": True,
        "del_modified_date": True,
        "del_user_data": True,
    },
}


SBML_TYPES_TO_NOT_ANONYMIZE = []


SBML_TYPES_TO_CANONICALIZE = [libsbml.SBML_SPECIES]


DEFAULT_AVAILABLE_PACKAGES = [
    "numpy",
    "pandas",
    "libsbml",
    "math",
    "scipy",
    "jax",
    "sklearn",
    "io",
    "traceback",
]


f = lambda t: dict(type=t)

SBML_GRAPH_PARAMS = dict(
    species=f("species"),
    reactions=f("reaction"),
    reactants=f("reactant"),
    products=f("product"),
    modifiers=f("modifier"),
)


MODEL_TO_API_KEY_NAME = {
    "gemini-2.5-pro-preview-03-25": "GEMINI_API_KEY",
    "claude-3-5-haiku-20241022": "CLAUDE_API_KEY",
    "claude-3-7-sonnet-20250219": "CLAUDE_API_KEY",
}


# DEMO_SBML_FILENAMES = [
#     "BIOMD0000000690",
#     "BIOMD0000000073",
#     "BIOMD0000000900",
#     "BIOMD0000000507",
#     "BIOMD0000000365",
#     "BIOMD0000000984",
#     "BIOMD0000000567",
#     "BIOMD0000000261",
#     "BIOMD0000000788",
#     "BIOMD0000000481",
# ]


# DEMO_SBML_FILENAMES = [
#     "BIOMD0000000328",
#     "BIOMD0000000051",
#     "BIOMD0000000465",
#     "BIOMD0000000709",
#     "BIOMD0000000228",
#     "BIOMD0000000336",
#     "BIOMD0000000835",
#     "BIOMD0000000026",
#     "BIOMD0000000013",
#     "BIOMD0000000934",
#     "BIOMD0000000434",
#     "BIOMD0000000169",
#     "BIOMD0000000586",
#     "BIOMD0000000411",
#     "BIOMD0000000857",
#     "BIOMD0000000446",
#     "BIOMD0000000106",
#     "BIOMD0000000610",
#     "BIOMD0000000578",
#     "BIOMD0000000658",
# ]


# TINY
# DEMO_SBML_FILENAMES = [
#     "BIOMD0000000777",
#     "BIOMD0000000054",
#     "BIOMD0000000657",
#     "BIOMD0000000845",
#     "BIOMD0000000319",
# ]

# SMALL
DEMO_SBML_FILENAMES = [
    "BIOMD0000001014",
    "BIOMD0000000929",
    "BIOMD0000000004",
    "BIOMD0000000708",
    "BIOMD0000000306",
    "BIOMD0000000984",
    "BIOMD0000000043",
    "BIOMD0000000962",
    "BIOMD0000000744",
    "BIOMD0000000609",
]

# MEDIUM
# DEMO_SBML_FILENAMES = ['BIOMD0000001004', 'BIOMD0000000827', 'BIOMD0000000757', 'BIOMD0000000438', 'BIOMD0000000530', 'BIOMD0000000082', 'BIOMD0000000080', 'BIOMD0000000041', 'BIOMD0000000415', 'BIOMD0000000586']


# LARGE (20)
# DEMO_SBML_FILENAMES = [
#     "BIOMD0000000832",
#     "BIOMD0000000467",
#     "BIOMD0000000315",
#     "BIOMD0000000209",
#     "BIOMD0000000143",
#     "BIOMD0000000940",
#     "BIOMD0000000660",
# ]
