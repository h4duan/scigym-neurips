"""Usage
python preprocess_models.py \
    --path_to_input /mfs1/u/stephenzlu/biomodels/curated/pass_qa/ \
    --path_to_output /mfs1/u/stephenzlu/biomodels/curated/pass_qa2/ \
    --overwrite \
    --stages 1 2 3 4
"""
import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from scigym.constants import (
    ANONYMIZE_EVERYTHING,
    ANONYMIZE_EVERYTHING_EXCEPT_SPECIES_CONFIG,
    DEMO_SBML_FILENAMES,
    SBML_TYPES_TO_CANONICALIZE,
    SBML_TYPES_TO_NOT_ANONYMIZE,
)
from scigym.sbml import SBML


def prepare_sbml_for_benchmark(sbml_raw: SBML, stages=[1, 2, 3, 4]):
    if 1 in stages:
        sbml_raw._remove_metadata(config=ANONYMIZE_EVERYTHING_EXCEPT_SPECIES_CONFIG)
    elif 1.5 in stages:
        sbml_raw._remove_metadata(config=ANONYMIZE_EVERYTHING)

    if 2 in stages:
        sbml_raw.shuffle_all()

    real_to_canonical_names: Dict[int, Dict[str, str]] = defaultdict(dict)
    real_to_fake_ids: Dict[int, Dict[str, str]] = defaultdict(dict)

    if 3 in stages:
        real_to_canonical_names = sbml_raw._canonicalize_names(
            type_codes_to_include=SBML_TYPES_TO_CANONICALIZE
        )
    if 4 in stages:
        real_to_fake_ids = sbml_raw._scramble_ids(type_codes_to_ignore=SBML_TYPES_TO_NOT_ANONYMIZE)
    return real_to_fake_ids, real_to_canonical_names


def main(args):
    xml_src_files = list(Path(args.path_to_input).glob("*.xml"))

    if isinstance(args.whitelist, list):
        xml_src_files = [x for x in xml_src_files if x.stem in args.whitelist]
    elif isinstance(args.blacklist, list):
        xml_src_files = [x for x in xml_src_files if x.stem not in args.blacklist]

    for xml_src_file in tqdm(xml_src_files):
        ########### STEP 1 - Remove all unnecessary metadata ###########
        sbml = SBML(str(xml_src_file))

        if "1" in args.stages:
            print(f"Processing {xml_src_file.stem} - Step 1: Removing metadata")
            sbml._remove_metadata(config=ANONYMIZE_EVERYTHING_EXCEPT_SPECIES_CONFIG)

        ########### STEP 2 - Shuffle the order of all components ###########
        if "2" in args.stages:
            print(f"Processing {xml_src_file.stem} - Step 2: Shuffling components")
            sbml.shuffle_all()

        ########### STEP 3 - Canonicalize the format of names ###########
        real_to_fake_ids: Dict[int, Dict[str, str]] = defaultdict(dict)
        real_to_canonical_names: Dict[int, Dict[str, str]] = defaultdict(dict)
        id_mapping_file = Path(args.path_to_output) / xml_src_file.name.replace(
            ".xml", "_id_mapping.json"
        )
        name_mapping_file = Path(args.path_to_output) / xml_src_file.name.replace(
            ".xml", "_name_mapping.json"
        )

        if "3" in args.stages:
            print(f"Processing {xml_src_file.stem} - Step 3: Canonicalizing names")
            real_to_canonical_names = sbml._canonicalize_names(
                type_codes_to_include=SBML_TYPES_TO_CANONICALIZE
            )

        ########### STEP 4 - Scramble the ids of relevant components ###########
        if "4" in args.stages:
            print(f"Processing {xml_src_file.stem} - Step 4: Scrambling ids")
            real_to_fake_ids = sbml._scramble_ids(type_codes_to_ignore=SBML_TYPES_TO_NOT_ANONYMIZE)

        # Save the processed SBML file to the output directory if it does not exist or overwrite is set
        output_file = Path(args.path_to_output) / xml_src_file.name
        if not output_file.exists() or args.overwrite:
            sbml.save(str(output_file))
            with open(id_mapping_file, "w") as f:
                json.dump(real_to_fake_ids, f, indent=4)
            with open(name_mapping_file, "w") as f:
                json.dump(real_to_canonical_names, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--path_to_input", required=True, help="Path to the input raw xml files")
    parser.add_argument(
        "--path_to_output", required=True, help="Path to the output processed files"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="List of stages to run, e.g. 1 2 3 4",
    )
    parser.add_argument("--overwrite", action="store_true", help="Should overwrite existing files")
    parser.add_argument(
        "--whitelist",
        nargs="+",
        default=DEMO_SBML_FILENAMES,
        help="List of filenames to only include",
    )
    parser.add_argument(
        "--blacklist",
        nargs="+",
        default=None,
        help="List of filenames to exclude, ignored if whitelist is used",
    )

    args = parser.parse_args()
    assert os.path.exists(args.path_to_input)
    os.makedirs(args.path_to_output, exist_ok=True)

    print(args)
    main(args)
