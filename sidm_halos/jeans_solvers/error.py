


class SIDMSolutionError(Exception):
    def __init__(self, msg, fit_result=None):
        super().__init__(msg)
        self.fit_result = fit_result
