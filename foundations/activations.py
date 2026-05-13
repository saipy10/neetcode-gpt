import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ans = []

        for x in z:
            p = 1 / (1 + np.e**(-x))
            ans.append(np.float64(np.round(p, 5)))
        return ans

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        ans = []
        for x in z:
            p = max(0, x)
            ans.append(np.float64(np.round(p, 5)))
        return ans
