"""Set of custom exception for FastFederatedLearning"""

from typing import Optional, LiteralString


class MutuallyExclusiveArgumentsException(Exception):
    """Exception raised when two mutually exclusive arguments are both specified."""

    def __init__(self,
                 message: Optional[LiteralString] = "Two or more mutually exclusive parameters have been specified."):
        """Exception raised when two mutually exclusive arguments are both specified.

        :param message: message to bring to the handler of the exception.
        :type message: Optional[str]
        """
        super().__init__(message)


class WronglySpecifiedArgumentException(Exception):
    """Exception raised when an argument is wrongly specified."""

    def __init__(self, message: Optional[LiteralString] = "At least one argument was wrongly specified."):
        """Exception raised when an argument is wrongly specified.

         :param message: message to bring to the handler of the exception.
         :type message: Optional[str]
         """
        super().__init__(message)
