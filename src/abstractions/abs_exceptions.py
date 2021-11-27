class ExportedExists(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class ConfigNotFound(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class FoundMultipleConfigs(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class ConfigParamDoesNotExist(Exception):
    pass
