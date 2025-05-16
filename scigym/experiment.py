from textwrap import dedent

from scigym.api import ExperimentConfig, ExperimentResult
from scigym.exceptions import ApplyExperimentActionError, ParseExperimentActionError
from scigym.sbml import SBML
from scigym.simulator import Simulator


class Experiment:
    """
    Base class for experiments.
    """

    def __init__(
        self,
        true_sbml: SBML,
        inco_sbml: SBML,
    ):
        """
        Initialize with a simulator and an SBML model.

        Args:
            simulator: The simulator to use for experiments
            sbml: The SBML model to experiment on
        """
        self.sbml = true_sbml
        self.allowed_functions = [
            "change_reaction_rate",
            "nullify_reaction",
            "change_initial_concentration",
            "nullify_species",
        ]
        self.whitelisted_species = inco_sbml.get_species_ids()
        self.whitelisted_parameters = inco_sbml.get_parameter_ids()
        self.whitelisted_reactions = inco_sbml.get_reaction_ids()

    def _check_allowed_functions(self, code):
        """
        Checks if each element in the list is calling an allowed function.

        Args:
            function_calls: A list of function call strings to check

        Returns:
            bool: True if all function calls are allowed, False otherwise
            list: List of invalid function calls if any exist
        """

        # Check each function call
        invalid_calls = []

        for call in code:
            # Extract the function name from the call
            if not isinstance(call, str):
                invalid_calls.append(f"Invalid type: {type(call)}, expected string")
                continue

            # Simple parsing to extract the function name
            # This assumes the function call is in the format: "function_name(args)"
            try:
                function_name = call.split("(")[0].strip()

                # Check if it's an allowed function
                if function_name not in self.allowed_functions:
                    invalid_calls.append(call)
            except:
                invalid_calls.append(call)

        # Return the result
        if invalid_calls:
            return False, invalid_calls
        else:
            return True, []

    def add_sbml_prefix(self, function_calls, indent_level=0):
        """
        Adds 'current_sbml.' prefix to each function call in the list and formats them
        as sequential statements in a single string with proper indentation.

        Args:
            function_calls: A list of function call strings
            indent_level: Number of indentation levels (default 0)

        Returns:
            str: A single string with all prefixed function calls as valid Python code
        """
        indent = " " * 4 * indent_level  # 4 spaces per indentation level
        result = []

        for call in function_calls:
            if not isinstance(call, str):
                continue

            # Simply add the prefix to the beginning of each call
            prefixed_call = f"{indent}current_sbml.{call}"
            result.append(prefixed_call)

        # Join all calls with newlines to create a sequence of statements
        return "\n".join(result)

    def call_simulator(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Abstract method to be implemented by subclasses.

        Args:
            config: Configuration for the experiment

        Returns:
            ExperimentResult containing experimental data
        """
        # Check for invalid species in the requested observation list
        wrong_species = set(config.observed_species) - set(self.whitelisted_species)
        if len(wrong_species) > 0:
            return ExperimentResult(
                success=False,
                error_message=f"Some of the species you requested to observe are not recognized: {wrong_species}",
            )

        # Check if the experiment actions can be successfully applied
        try:
            actions = config.experiment_action
            if len(actions) > 0:
                current_sbml = SBML.apply_experiment_actions(
                    sbml=self.sbml,
                    experiment_actions=actions,
                    valid_species_ids=self.whitelisted_species,
                    valid_reaction_ids=self.whitelisted_reactions,
                )
            else:
                current_sbml = self.sbml
        except ParseExperimentActionError as e:
            return ExperimentResult(
                success=False,
                error_message="We were not able to run the experiment with your set experiment actions. "
                + str(e),
            )
        except ApplyExperimentActionError as e:
            return ExperimentResult(
                success=False,
                error_message="We were not able to run the experiment with your set experiment actions. "
                + str(e),
            )

        # Check for simulator errors when running the experiment
        try:
            simulation = Simulator(current_sbml)
            return simulation.run(
                observed_species=config.observed_species,
                rm_concentration_brackets=True,
            )
        except Exception as e:
            print(f"Error during simulation of experiment: {e}")
            return ExperimentResult(
                success=False,
                error_message=dedent(
                    """
                We were not able to run the experiment with your set experiment actions.
                Please scrutinize your protocol and make sure that the experiment you request is sensical.
                """
                ).strip(),
            )
