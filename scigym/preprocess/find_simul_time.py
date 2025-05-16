import math
from typing import Callable, Tuple

import libsedml
import roadrunner as rr
from tqdm import tqdm

from scigym.sbml import SBML
from scigym.simulator import Simulator


def extract_time_from_error(error_msg):
    import re

    # Look for "At t = X" pattern where X is a number
    match = re.search(r"At t = ([\d.]+)", error_msg)

    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return None

    return None


def binary_search(low: float, high: float, oracle: Callable[[float], bool]) -> float:
    """
    Perform a binary search to find the time at which the simulation is stable.

    :param low: The lower bound of the search interval.
    :param high: The upper bound of the search interval.
    :param oracle: A function that takes a time value and returns True if the simulation is stable at that time, False otherwise.
    :return: The time at which the simulation is stable.
    """
    while low < high:
        mid = (low + high) / 2
        if oracle(mid):
            high = mid
        else:
            low = mid + 1
    return low


def line_search(low: float, high: float, oracle: Callable) -> Tuple[float, int]:
    converged = False
    n_steps = 0
    best_n_steps = 0
    best_time = low
    current_time = low
    max_rate_of_change = float("inf")

    while not converged and current_time < high:
        # Simulate for another time step
        converged, failed, current_time, max_roc = oracle(current_time)
        n_steps += 1

        if converged:
            print(f"System converged at time {current_time}")
            best_time = current_time
            break

        if failed:
            print(f"Simulation failed at time {current_time}")
            break

        # print(f"Current time: {current_time}, did not converge yet")

        if max_roc < max_rate_of_change:
            max_rate_of_change = max_roc
            best_time = current_time
            best_n_steps = n_steps

    if math.isinf(best_time):
        best_time = low

    return best_time, best_n_steps


class SimulationTimeFinder:
    def __init__(
        self,
        simulator: Simulator,
        max_end_time=10000,
        max_num_steps=10000,
        step_size=0.1,
    ):
        self.simulator = simulator
        self.max_end_time = max_end_time
        self.max_num_steps = max_num_steps
        self.step_size = step_size

    def oracle(self, time: float) -> bool:
        self.simulator.updateSimulationParameters(outputEndTime=time)
        try:
            self.simulator.run()
        except Exception as e:
            print(f"Simulation failed at time {time}: {e}")
            return False

        # Check if the simulation has reached a steady state
        max_rate_of_change = max(self.simulator._rr.getRatesOfChange())
        if max_rate_of_change > 1e-6:
            print(f"Simulation did not converge at time {time}")
            return False
        return True

    def all_steps(self, time: float) -> Tuple[bool, bool, float, float]:
        converged, failed = False, False
        self.simulator.updateSimulationParameters(outputEndTime=time)

        try:
            self.simulator.run()
        except Exception as e:
            failed = True

        new_time = self.simulator._rr.getCurrentTime() + self.simulator._rr.getDiffStepSize()

        rates = self.simulator._rr.getRatesOfChange()
        max_rate_of_change = max(rates)
        if max_rate_of_change < 10e-6:
            converged = True

        return converged, failed, new_time, max_rate_of_change

    def one_step(self, *args, **kwargs) -> Tuple[bool, bool, float, float]:
        converged, failed = False, False
        currentTime = self.simulator._rr.getCurrentTime()
        # integrator: rr.Integrator = self.simulator._rr.getIntegrator()
        stepSize = self.simulator._rr.getDiffStepSize()

        try:
            new_time = self.simulator._rr.oneStep(currentTime, stepSize, reset=False)
            # new_time = currentTime + self.step_size
            # new_time = currentTime + stepSize
            # self.simulator._rr.simulate(start=currentTime, end=None)
            assert self.simulator._rr.getCurrentTime() == new_time
        except Exception as e:
            failed = True
            new_time = currentTime

        rates = self.simulator._rr.getRatesOfChange()
        max_rate_of_change = max(rates) if len(rates) > 0 else -float("inf")
        if max_rate_of_change < 10e-6:
            converged = True

        if not all(self.simulator.event_tracker.values()):
            untriggered_events = [
                e for e in self.simulator.event_tracker if not self.simulator.event_tracker[e]
            ]
            print(f"Untriggered events: {untriggered_events}, keep searching...")
            converged = False

        return converged, failed, new_time, max_rate_of_change

    def update_simulation_time(self, time: float, steps: int | None = None) -> None:
        time = math.floor(time)
        sid = self.simulator.simulation.getId()
        self.simulator.updateSimulationParameters(outputEndTime=time)
        if steps is not None:
            self.simulator.updateSimulationParameters(numberOfSteps=steps)
        simulation: libsedml.SedUniformTimeCourse = (
            self.simulator.sbml.sedml_document.getSimulation(sid)
        )
        assert simulation.setOutputEndTime(time) == libsedml.LIBSEDML_OPERATION_SUCCESS
        if steps is not None:
            assert simulation.setNumberOfSteps(steps) == libsedml.LIBSEDML_OPERATION_SUCCESS
        self.simulator.sbml.sed_simulation = simulation.clone()

    def save_sedml(self, path: str) -> int:
        return self.simulator.sbml.save_sedml(path)

    def find_simulation_time(self) -> Tuple[float, int]:
        """
        Find the time at which the simulation is stable.

        :return: The time at which the simulation is stable.
        """
        initial_parameters = self.simulator.getSimulationParameters()
        output_end_time = initial_parameters["outputEndTime"]
        number_of_steps: int = initial_parameters["numberOfSteps"]  # type: ignore
        print(f"Original simulation : {output_end_time}, {number_of_steps}")

        # Check if we can directly solve the simulation using steady state analysis
        steady_state_time = -1
        try:
            steady_state = self.simulator._rr.steadyState()
            if steady_state < 10e-6:
                self.simulator._rr.steadyStateSelections = ["time"]
                steady_state_time = self.simulator._rr.getSteadyStateValues()[0]
                print(f"Steady state reached with value {steady_state} at time {steady_state_time}")
        except Exception as e:
            print(f"Steady state analysis failed: {e}")

        max_end_time = max(self.max_end_time, output_end_time)
        min_num_steps = min(self.max_num_steps, number_of_steps)

        self.simulator.prepare_simulation()
        # self.simulator._rr.simulate(end=output_end_time, points=2)
        linear_search_time, linear_n_steps = line_search(
            output_end_time, max_end_time, self.one_step
        )

        print(f"Linear search: {linear_search_time}, {linear_n_steps}")

        if linear_search_time > self.max_end_time and output_end_time > 10.0:
            best_time = output_end_time
            best_n_steps = number_of_steps
        else:
            best_time = max(linear_search_time, steady_state_time, output_end_time)
            linear_n_steps = min(linear_n_steps, self.max_num_steps)
            best_n_steps = max(linear_n_steps, min_num_steps)

        print(f"Final chosen: {best_time}, {best_n_steps}")
        print()
        return best_time, best_n_steps


if __name__ == "__main__":
    import os
    from pathlib import Path

    sbml_files = list(Path("/mfs1/u/stephenzlu/biomodels/curated/pass_qa").glob("*.xml"))
    problem_files = []

    # sbml_files = [Path("/mfs1/u/stephenzlu/biomodels/curated/pass_qa/BIOMD0000001042.xml")]

    for sbml_file in tqdm(sbml_files):
        sedml_file = sbml_file.with_suffix(".sedml")
        assert sbml_file.exists()
        assert sedml_file.exists()

        target_path = str(sedml_file).replace("pass_qa", "new_sedml")

        if os.path.exists(target_path):
            continue

        try:
            sbml = SBML(str(sbml_file), str(sedml_file))
            simulator = Simulator(sbml=sbml)
        except Exception as e:
            print(e)
            problem_files.append(sbml_file)
            continue

        # Find the best simulation end time
        finder = SimulationTimeFinder(simulator)
        best_time, best_nsteps = finder.find_simulation_time()
        best_time = math.floor(best_time)

        # Update the simulation time in the SBML
        finder.update_simulation_time(best_time, best_nsteps)

        new_simulator = Simulator(sbml=finder.simulator.sbml)
        new_finder = SimulationTimeFinder(new_simulator)
        params = new_finder.simulator.getSimulationParameters()
        assert params["outputEndTime"] == best_time
        assert params["numberOfSteps"] == best_nsteps

        # Check failure
        log_path = f"/tmp/{sbml_file.stem}.log"
        rr.Logger_disableConsoleLogging()
        rr.Logger.enableFileLogging(log_path, rr.Logger.LOG_ERROR)

        succeeded = False
        fail_time = None
        try:
            data = new_finder.simulator.run()
            assert data.result is not None
            succeeded = True
        except Exception as e:
            if os.path.exists(log_path):
                with open(log_path, "r") as log_file:
                    log_content = log_file.read()
                print(log_content)
                fail_time = extract_time_from_error(log_content)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

        # succeeded = False
        # fail_time = None

        # f = io.StringIO()

        # try:
        #     with redirect_stderr(f):
        #         data = new_finder.simulator.run()
        #         assert data.result is not None
        #         succeeded = True
        # except Exception as e:
        #     stderr_output = f.getvalue()
        #     print(stderr_output)
        #     fail_time = extract_time_from_error(stderr_output)
        #     print(f"Failed at time: {fail_time}")
        # finally:
        #     f.close()

        if fail_time is not None:
            finder.update_simulation_time(fail_time)
            new_simulator = Simulator(sbml=finder.simulator.sbml)
            new_finder = SimulationTimeFinder(new_simulator)
            try:
                data = finder.simulator.run()
                assert data.result is not None
                succeeded = True
            except Exception as e:
                print("Failed again")
                print(e)
                problem_files.append(sbml_file)
                continue

        print(new_finder.simulator.getSimulationParameters())

        # if succeeded:
        #     new_finder.save_sedml(target_path)

    print(len(problem_files))
    print(problem_files)
