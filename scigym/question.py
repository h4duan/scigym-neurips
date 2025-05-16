import os

from scigym.sbml import SBML


class Question:
    """
    Class for creating a research context prompt
    """

    research_question: str
    task_difficulty: str
    original_sbml_model: SBML
    partial_sbml_model: SBML
    partial_runnable_sbml_model: SBML

    def __init__(
        self,
        sbml_directory_path: str,
        task_difficulty: str,
    ):
        """
        Initialize with sbml_model and task difficulty choice

        Args:
            sbml_model_path: A path to an SBML xml file
            difficulty_level: Benchmark task difficulty
            path_to_question_yml: A path to a yaml config file specifying a question
        """
        self.task_difficulty = task_difficulty
        self.sbml_directory_path = sbml_directory_path
        self.load()

    def get_research_question(self) -> str:
        assert self.research_question is not None
        return self.research_question

    def get_partial_sbml(self) -> SBML:
        assert self.partial_sbml_model is not None
        return self.partial_sbml_model

    def get_runnable_partial_sbml(self) -> SBML:
        assert self.partial_runnable_sbml_model is not None
        return self.partial_runnable_sbml_model

    def get_original_sbml(self) -> SBML:
        assert self.original_sbml_model is not None
        return self.original_sbml_model

    def create_research_question(self) -> str:
        """
        Create a research question based on the masked SBML model and difficulty level.

        Returns:
            A research question as a string
        """
        raise NotImplementedError()

    def create_partial_sbml_from_commands(self) -> SBML:
        """
        Create an incomplete SBML model by executing the commands provided by an LLM
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the question directly from a yaml file that contains paths to the research question,
        ground truth sbml file, simulation config sedml file, partially masked sbml file, and
        additional metadata required to setup a full benchmark instance after preprocessing
        """

        # Access the paths through our dataclass structure
        path_to_truth_sbml = f"{self.sbml_directory_path}/truth.xml"
        path_to_truth_sedml = f"{self.sbml_directory_path}/truth.sedml"
        path_to_partial_sbml = f"{self.sbml_directory_path}/partial.xml"
        path_to_partial_runnable_sbml = f"{self.sbml_directory_path}/partial.xml"

        assert os.path.exists(path_to_truth_sedml)
        assert os.path.exists(path_to_truth_sbml)
        assert os.path.exists(path_to_partial_sbml)
        assert os.path.exists(path_to_partial_runnable_sbml)
        # assert os.path.exists(path_to_question)

        try:
            self.original_sbml_model = SBML(path_to_truth_sbml, path_to_truth_sedml)
            self.partial_sbml_model = SBML(path_to_partial_sbml, path_to_truth_sedml)
            self.partial_runnable_sbml_model = SBML(
                path_to_partial_runnable_sbml, path_to_truth_sedml
            )

        except Exception as e:
            raise ValueError(f"Failed to read question from {self.sbml_directory_path}: {str(e)}")
