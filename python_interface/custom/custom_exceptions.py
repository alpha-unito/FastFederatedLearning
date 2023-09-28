"""Set of custom exception for FastFederatedLearning"""


class MutuallyExclusiveArgumentsException(Exception):
    """Exception raised when two mutually exclusive arguments are both specified.

    :param message: message to bring to the handler of the exception.
    :type message: Optional[str]
    """

    def __init__(self, message="Two or more mutually exclusive parameters have been specified."):
        self.message = message
        super().__init__(self.message)
