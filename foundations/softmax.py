import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxi = max(z)
        e = np.e
        probabilities = []
        denominator = sum(e**(i-maxi) for i in z)
        for i in z:
            probabilities.append(np.round((e**(i-maxi))/denominator, 4))

        return probabilities 
