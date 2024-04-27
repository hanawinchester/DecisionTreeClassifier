package components.decisiontree;
/**
 * interface for a decision tree classifier kernel
 * 
 * @param <T> the type of features used in the classifier
 * @param <U> target labels
 */
public interface DecisionTreeClassifierKernel<T, U> {

    /**
     * trains the decision tree classifier using the input features X and target labels y
     * 
     * @param X The input features for training
     * @param y The target labels for training
     * @requires X and y are not null, X.length == y.length, and X.length > 0
     * @ensures isTrained(this.state)
     */
    void fit(double[][] X, int[] y);

    /**
     * predicts the class labels for a new set of input features X using the trained decision tree
     * 
     * @param X the new set of input features for prediction
     * @return predicted class labels.
     * @requires isTrained(this.state) and X is not null
     */
    Integer[] predict(double[][] X);

    /**
     * returns depth of the decision tree
     * 
     * @return the depth of decision tree
     * @requires isTrained(this.state)
     */
    int getDepth();

}
