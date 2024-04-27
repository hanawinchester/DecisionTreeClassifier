package components.decisiontree;
//extends kernel

/**
 * interface for enhanced decision tree classifier
 * 
 * @param <T> the type of features used in the classifier
 * @param <U> target labels used in the classifier
 */
public interface DecisionTreeClassifierEnhanced<T, U> extends DecisionTreeClassifierKernel<T, U> {

    /**
     * clears the decision tree classifier, resetting it to an initial state.
     */
    void clear();

    /**
     * creates new instance of the decision tree classifier
     * 
     * @return a new instance of the decision tree classifier
     */
    DecisionTreeClassifierEnhanced<T, U> newInstance();

    /**
     * returns depth of the decision tree
     * 
     * @return the depth of decision tree
     * @requires isTrained(this.state)
     */
    int getDepth();

}
