import libsbml


def get_plugins_from_sbml(sbml_file_path):
    """
    Extract detailed information about all plugins being used in an SBML file.

    Args:
        sbml_file_path (str): Path to the SBML file

    Returns:
        dict: Detailed information about all plugins
    """
    # Read the SBML file
    reader = libsbml.SBMLReader()
    document: libsbml.SBMLDocument = reader.readSBML(sbml_file_path)

    # Check for errors
    if document.getNumErrors() > 0:
        print(f"Error reading SBML file: {document.getError(0).getMessage()}")
        return None

    # Get the model
    model = document.getModel()
    if model is None:
        print("No model found in the SBML file")
        return None

    # Dictionary to store plugin details
    plugins_info = {}

    # Get all registered package names
    extension_registry: libsbml.SBMLExtensionRegistry = libsbml.SBMLExtensionRegistry.getInstance()
    registered_packages = [
        extension_registry.getRegisteredPackageName(j)
        for j in range(extension_registry.getNumRegisteredPackages())
    ]
    print(registered_packages)

    # Function to extract plugin information
    def extract_plugin_info(element, element_type, element_id):
        for pkg_name in registered_packages:
            plugin: libsbml.SBasePlugin = element.getPlugin(pkg_name)
            if plugin is not None:
                plugin_type_name = plugin.getElementNamespace()

                if pkg_name not in plugins_info:
                    plugins_info[pkg_name] = []

                plugin_info = {
                    "plugin_type": plugin_type_name,
                    "element_type": element_type,
                    "element_id": element_id,
                    "plugin_object": plugin.__class__.__name__,
                }

                # Try to get more detailed information based on package type
                if pkg_name == "layout":
                    if hasattr(plugin, "getNumLayouts"):
                        plugin_info["num_layouts"] = plugin.getNumLayouts()  # type: ignore
                elif pkg_name == "fbc":
                    if hasattr(plugin, "getChemicalFormula"):
                        plugin_info["has_chemical_formula"] = bool(plugin.getChemicalFormula())  # type: ignore
                    if hasattr(plugin, "getCharge"):
                        plugin_info["has_charge"] = plugin.isSetCharge()  # type: ignore
                    if hasattr(plugin, "getLowerFluxBound") and hasattr(
                        plugin, "getUpperFluxBound"
                    ):
                        plugin_info["has_flux_bounds"] = plugin.isSetLowerFluxBound() or plugin.isSetUpperFluxBound()  # type: ignore
                elif pkg_name == "groups":
                    if hasattr(plugin, "getNumGroups"):
                        plugin_info["num_groups"] = plugin.getNumGroups()  # type: ignore

                plugins_info[pkg_name].append(plugin_info)

    # Check document-level plugins
    for pkg_name in registered_packages:
        doc_plugin: libsbml.SBasePlugin = document.getPlugin(pkg_name)
        if doc_plugin is not None:
            plugin_type_name = doc_plugin.getElementNamespace()
            if pkg_name not in plugins_info:
                plugins_info[pkg_name] = []
            plugins_info[pkg_name].append(
                {
                    "plugin_type": plugin_type_name,
                    "element_type": "document",
                    "element_id": "document",
                    "plugin_object": doc_plugin.__class__.__name__,
                }
            )

    # Check model-level plugins
    extract_plugin_info(model, "model", model.getId() or "model")

    # Check compartments
    for i in range(model.getNumCompartments()):
        compartment = model.getCompartment(i)
        extract_plugin_info(compartment, "compartment", compartment.getId() or f"compartment_{i}")

    # Check species
    for i in range(model.getNumSpecies()):
        species = model.getSpecies(i)
        extract_plugin_info(species, "species", species.getId() or f"species_{i}")

    # Check reactions
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        extract_plugin_info(reaction, "reaction", reaction.getId() or f"reaction_{i}")

        # Also check kinetic law if present
        if reaction.isSetKineticLaw():
            kinetic_law = reaction.getKineticLaw()
            extract_plugin_info(
                kinetic_law, "kineticLaw", f"kineticLaw_of_{reaction.getId() or f'reaction_{i}'}"
            )

    # Check parameters (global)
    for i in range(model.getNumParameters()):
        parameter = model.getParameter(i)
        extract_plugin_info(parameter, "parameter", parameter.getId() or f"parameter_{i}")

    # Check rules
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        extract_plugin_info(rule, "rule", rule.getVariable() or f"rule_{i}")

    # Check constraints
    for i in range(model.getNumConstraints()):
        constraint = model.getConstraint(i)
        extract_plugin_info(constraint, "constraint", f"constraint_{i}")

    # Check events
    for i in range(model.getNumEvents()):
        event = model.getEvent(i)
        extract_plugin_info(event, "event", event.getId() or f"event_{i}")

        # Check event assignments
        for j in range(event.getNumEventAssignments()):
            assignment = event.getEventAssignment(j)
            extract_plugin_info(
                assignment,
                "eventAssignment",
                f"eventAssignment_{assignment.getVariable()}_of_{event.getId() or f'event_{i}'}",
            )

    # Check function definitions
    for i in range(model.getNumFunctionDefinitions()):
        function_def = model.getFunctionDefinition(i)
        extract_plugin_info(
            function_def, "functionDefinition", function_def.getId() or f"functionDefinition_{i}"
        )

    # Check initial assignments
    for i in range(model.getNumInitialAssignments()):
        init_assignment = model.getInitialAssignment(i)
        extract_plugin_info(
            init_assignment,
            "initialAssignment",
            init_assignment.getSymbol() or f"initialAssignment_{i}",
        )

    # Check units definitions
    for i in range(model.getNumUnitDefinitions()):
        unit_def = model.getUnitDefinition(i)
        extract_plugin_info(unit_def, "unitDefinition", unit_def.getId() or f"unitDefinition_{i}")

    return plugins_info


def print_plugin_details(plugins_info):
    """Print detailed information about plugins in a readable format"""
    if not plugins_info:
        print("No plugins found or error reading the file.")
        return

    print("\n=== DETAILED SBML PLUGINS INFORMATION ===\n")

    for pkg_name, plugins in plugins_info.items():
        print(f"\n{pkg_name.upper()} PACKAGE PLUGINS:")
        print("-" * 50)

        # Group plugins by type
        plugin_types = {}
        for plugin in plugins:
            plugin_type = plugin["plugin_type"]
            if plugin_type not in plugin_types:
                plugin_types[plugin_type] = []
            plugin_types[plugin_type].append(plugin)

        # Print each plugin type
        for plugin_type, instances in plugin_types.items():
            print(f"\n  Plugin Type: {plugin_type}")
            print(f"  Implementation: {instances[0]['plugin_object']}")
            print(f"  Used in {len(instances)} element(s):")

            for instance in instances:
                print(f"    - {instance['element_type']}: {instance['element_id']}")

                # Print additional plugin-specific information if available
                for key, value in instance.items():
                    if key not in ["plugin_type", "element_type", "element_id", "plugin_object"]:
                        print(f"      {key}: {value}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_sbml_plugins.py <sbml_file.xml>")
        sys.exit(1)

    sbml_file = sys.argv[1]
    plugins = get_plugins_from_sbml(sbml_file)
    print_plugin_details(plugins)
