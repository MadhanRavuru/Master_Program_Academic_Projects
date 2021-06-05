"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    R = y.shape[0]                                  #examples
    D,C = W.shape                                   #dimension,classes
    output=np.zeros(C)
    prob=np.zeros(C)
    for i in range(R):
        for j in range(C):
            output[j]=np.dot(X[i],W[:,j])
        maxi=output.max(0)
        for j in range(C):
            output[j]=output[j]-maxi              #for stability
            prob[j]=np.exp(output[j])
        prob/=np.sum(prob)                        #softmax function probabilities
       
        loss-=np.log(prob[y[i]]) 
        
        prob[y[i]]-=1;
        for c in range(C):
            dW[:,c] += X[i,:] * prob[c]
            
    loss/=R 
    dW/=R
    #Regularization
    loss += 0.5*reg * np.sum(W ** 2)
    dW += reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num=X.shape[0]
    P=np.dot(X,W)              
    P=P-np.reshape(P.max(1),(num,1))      #subtracting each row by max of that row,to restrict the exp from growing,stability
    exp=np.exp(P)/np.sum(np.exp(P),axis=1,keepdims=True)   #500x1 bcoz of keepdims for sum, axis=1 row sum
    py=np.zeros((num,10))
    py[range(num),y]=1.0                                       #500x10 matrix with each row has class label
    loss=-np.sum(py*np.log(exp))/num + 0.5*reg*np.sum(W**2)
    dw=(exp-py)/num
    dW=np.dot(X.T,dw)+reg*W
  
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    #learning_rates = [1e-7, 5e-7]
    #regularization_strengths = [2.5e4, 5e4]
    
    learning_rates = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    regularization_strengths = [2.5e4, 5e4,1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    total = len(learning_rates) * len(regularization_strengths)
    iter = 0

    for l in learning_rates:
        for r in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=l, reg=r, num_iters=5000)
            
            y_train_pred = softmax.predict(X_train)
            y_train_acc = np.mean(y_train == y_train_pred)

            y_val_pred = softmax.predict(X_val)
            y_val_acc = np.mean(y_val == y_val_pred)

            results[l, r] = (y_train_acc, y_val_acc)
            if y_val_acc > best_val:
                best_val = y_val_acc
                best_softmax = softmax
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
