# Laplacian Smoothing
```math
P(w|c) 
= \dfrac{\text{count}(w, c) + 1}{\text{total count in class } c + V}
```

# Eigenvalues, Eigenvectors
```python
import numpy as np
A = np.array([[a, b], [c, d]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```

# Naive Bayes (TODO)
## Naive Bayes Classifier for Categorical Data
```python
from sklearn.naive_bayes import MultinomialNB
```

## Naive Bayes Classifier for Continuous Data
```python
from sklearn.naive_bayes import GaussianNB
```