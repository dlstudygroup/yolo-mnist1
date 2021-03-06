"""Custom exceptions."""


class TooSmallInputShape(Exception):
    """Exception raised when input shape is too small.

    Attributes:
        minimum_input_shape - can't be smaller than that
    """

    def __init__(self, input_shape, minimum_input_shape):
        self.minimum_input_shape = minimum_input_shape
        self.input_shape = input_shape

    def __str__(self):
        return f"Input shape {self.input_shape} should be \
        equal or greater than {self.minimum_input_shape}."
