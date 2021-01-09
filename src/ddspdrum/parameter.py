"""
Parameter Class
"""

import numpy as np


class Parameter:
    """
    Parameter class is a structure for keeping track of parameters
    that have a specific range. Also handles functionality for converting
    to and from a range between 0 and 1.

    Parameters
    ----------
    name    (str)   :   Unique name to give to this parameter.
    value   (float) :   initial value of this parameter
    minimum (float) :   minimum value that this parameter can take on
    maximum (float) :   maximum value that this parameter can take on
    scale   (float) :   scaling to apply when converting to and from a value with
                        range [0,1]. Defaults to 1 which is linear scaling. A value
                        less than 1 is a logarithmic relationship that fills more of the
                        lower range of the parameter, whereas a scale value greater 1
                        is an exponential relationship that fills more of the higher
                        range of the parameter.
    """

    def __init__(
        self,
        name: str,
        value: float,
        minimum: float,
        maximum: float,
        scale: float = 1,
    ):
        self.name = name
        self.value = np.clip(value, minimum, maximum)
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale

    def __str__(self):
        name = "{}, ".format(self.name) if self.name else ""
        return "Parameter: {}, Value: {}, Min: {}, Max: {}, Scale: {}".format(
            name, self.value, self.minimum, self.maximum, self.scale
        )

    def set_value(self, new_value):
        """
        Set value of this parameter clipped to the minimum and maximum range

        Parameters
        ---------
        new_value (float)   :   value to update parameter with
        """
        self.value = np.clip(new_value, self.minimum, self.maximum)

    def set_value_0to1(self, new_value):
        """
        Set value of this parameter using a normalized value in the range [0,1]

        Parameters
        ----------
        new_value (float)   :   value to update parameter with, in the range [0,1]
        """
        new_value = np.clip(new_value, 0, 1)

        if new_value != 0 and self.scale != 1:
            new_value = np.exp2(np.log2(new_value) / self.scale)

        self.value = self.minimum + (self.maximum - self.minimum) * new_value

    def get_value_0to1(self) -> float:
        """
        Get value of this parameter using a normalized value in range [0,1]
        """
        value = (self.value - self.minimum) / (self.maximum - self.minimum)
        if self.scale != 1:
            value = np.power(value, self.scale)

        return value