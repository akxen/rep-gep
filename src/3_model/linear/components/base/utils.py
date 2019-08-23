"""Utility class used to perform common calculations / simple operations"""


class Utilities:

    @staticmethod
    def get_absolute_difference(d1, d2):
        """Compute absolute difference for same keys between two dictionaries"""

        # Check that keys are the same
        assert set(d1.keys()) == set(d2.keys()), f'Keys are not the same: {d1.keys()}, {d2.keys()}'

        # Absolute difference for each key
        abs_difference = {k: abs(abs(d1[k]) - abs(d2[k])) for k in d1.keys()}

        return abs_difference

    def get_non_zero_absolute_difference(self, d1, d2):
        """Get elements with non-zero absolute difference between two dictionaries"""

        abs_difference = self.get_absolute_difference(d1, d2)

        return {k: v for k, v in abs_difference.items() if v > 0.01}

    def get_max_absolute_difference(self, d1, d2):
        """Compute max absolute difference between two dictionaries"""

        abs_difference = self.get_absolute_difference(d1, d2)

        # Max absolute difference
        max_abs_difference = max(abs_difference.values())

        return max_abs_difference
