package components.decisiontree;

import java.util.Objects;

import javax.swing.tree.TreeNode;

import components.binarytree.BinaryTree;
import components.binarytree.BinaryTree1;

public abstract class DecisionTreeClassifierSecondary<T, U> implements DecisionTreeClassifierEnhanced<T, U> {

    public BinaryTree<TreeNode> root;

    public void fit(double[][] X, int[] y) {
        this.root = new BinaryTree1<>();
        TreeNode rootNode = buildTree(X, y, 3); //build the tree
        this.root.assemble(rootNode, root, root);
    }

    protected abstract TreeNode buildTree(double[][] x, int[] y, int depthLimit);

    public Integer[] predict(double[][] X) {
        if (this.root == null) {
            throw new IllegalStateException("Tree isn't trained, use the fit method first.");
        }
        int size = X.length;
        Integer[] predictions = new Integer[size];
        for (int i = 0; i < size; i++) {
            predictions[i] = (Integer) predictSingleNode((DecisionTreeOnBinaryTree.TreeNode) root, X[i]);
        }
        return predictions;
    }

    protected abstract U predictSingleNode(DecisionTreeOnBinaryTree.TreeNode root2, double[] instance);

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof DecisionTreeClassifierSecondary)) {
            return false;
        }
        //no type parameter
        DecisionTreeClassifierSecondary<?, ?> other = (DecisionTreeClassifierSecondary<?, ?>) obj;
        return Objects.equals(this.root, other.root);
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.root);
    }

    @Override
    public String toString() {
        return "DecisionTreeClassifierSecondary[root=" + this.root + "]";
    }

    //will maybe delete this for simplicity
    @Override
    public int getDepth() {
        if (this.root == null) {
            throw new IllegalStateException("Tree isn't trained, use the fit method first.");
        }
        //calculate tree depth
        return 0;
    }
}
