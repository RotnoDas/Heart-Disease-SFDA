import importlib
import inspect
import pkgutil

import heart_sfda.adaptation as adaptation_pkg


FORBIDDEN_PARAMETERS = {
    "y",
    "y_true",
    "target_y",
    "y_target",
    "target_labels",
    "target_label",
    "ground_truth",
}


def test_public_adaptation_functions_do_not_accept_target_labels():

    violations = []

    for module_info in pkgutil.iter_modules(
        adaptation_pkg.__path__
    ):

        module_name = (
            f"{adaptation_pkg.__name__}."
            f"{module_info.name}"
        )

        module = importlib.import_module(
            module_name
        )

        for name, obj in inspect.getmembers(
            module,
            inspect.isfunction,
        ):

            # Ignore imported functions
            if (
                obj.__module__
                != module.__name__
            ):
                continue

            # Only inspect public adaptation APIs
            if name.startswith("_"):
                continue

            if "adapt" not in name.lower():
                continue

            signature = (
                inspect.signature(obj)
            )

            parameter_names = {
                parameter.lower()
                for parameter
                in signature.parameters
            }

            forbidden_found = (
                parameter_names
                &
                FORBIDDEN_PARAMETERS
            )

            if forbidden_found:

                violations.append(
                    {
                        "function":
                            (
                                f"{module.__name__}."
                                f"{name}"
                            ),

                        "forbidden":
                            sorted(
                                forbidden_found
                            ),
                    }
                )

    assert not violations, (
        "Target-label parameters found "
        "in public SFDA APIs:\n"
        f"{violations}"
    )