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
        super().__init__("The wanted tag does not exist in the local model registry")


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
            "Query model <%s> does not part in the available mapping, please consult /src/etl/model/mapping.py",
            query_model_name,
        )
