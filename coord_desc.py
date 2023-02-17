import numpy as np
from sklearn.preprocessing import StandardScaler

def logistic_regression(X, y, num_iterations=1000, tol=1e-4):
    # Initialize coefficients to zero
    w = np.zeros(X.shape[1])
    
    # Iterate until convergence or maximum number of iterations is reached
    for t in range(num_iterations):
        w_prev = np.copy(w)
        
        # Loop over coordinates and update weights
        for i in range(X.shape[1]):
            # Compute gradient of loss with respect to ith coordinate
            grad_i = np.dot(X[:,i], y - 1 / (1 + np.exp(-np.dot(X, w))))
            
            # Update ith coordinate
            w[i] += grad_i / np.dot(X[:,i], X[:,i])
        
        
        loss = log_loss(X,y)
        print(loss)
        # Check for convergence
        if np.linalg.norm(w - w_prev) < tol:
            break
    
    # Return final coefficients
    return w


def log_loss(X, y):
  # Compute the logistic loss
  y_pred = 1 / (1 + np.exp(-X.dot(self.coef_)))
  y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
  return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))



scaler = StandardScaler()

X = scaler.fit_transform(x)
y=y

final_weight = logistic_regression(X, y, num_iterations=1000, tol=1e-4)


