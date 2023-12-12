class UndetectedMonitor(Exception):
    def __init__(self):
        super().__init__("No monitor detected, screen capture can't be processed")


class UndetectedPrimaryMonitor(Exception):
    def __init__(self):
        super().__init__(
            "No primary monitor detected, screen capture can't be processed"
        )


class ExpectedClassfierVersionDoesNotExist(Exception):
    def __init__(self, tag):
        super().__init__(
            f"The wanted tag <{tag}> does not exist in the local model registry"
        )


class UnfoundClassifier(Exception):
    def __init__(self):
        super().__init__("No Classifier found")


class RegistryDoesNotExist(Exception):
    def __init__(self):
        super().__init__(
            "No registry found, expected to be at: /this-is-not-real-aerial-imagery/src/etl/model/registry"
        )


class ModelArchitectureUnvailable(Exception):
    def __init__(self, query_model_name: str):
        super().__init__(
            f"Query model <{query_model_name}> does not part in the available mapping, please consult /src/etl/model/mapping.py",
        )


class MissingConfigParameters(Exception):
    def __init__(self, parameter_name: str):
        super().__init__(
            f"Missing parameter in the config file <{parameter_name}>, please refer to the template to see what is required"
        )


class ExperiementRequired(Exception):
    def __init__(self):
        super().__init__(
            "Missing experiement name as use classifier where used",
        )


class BatchSizeCantBeZeroOrNegatif(Exception):
    def __init__(self):
        super().__init__(
            "Unaccepted value of batch size, it shoulb be a positif integer",
        )
