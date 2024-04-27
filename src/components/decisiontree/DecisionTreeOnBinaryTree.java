package components.decisiontree;

import components.binarytree.BinaryTree;
import components.binarytree.BinaryTree1;

import java.util.Objects;

import javax.swing.tree.TreeNode;

import java.util.Iterator;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Comparator;

@SuppressWarnings("unused")

/**
 * Convention and Correspondence:
 * 
 * Convention (the representation invariant):
 * -the decision tree represented by this class maintains the binary tree structure
 *   where each node has a feature, a threshold value, and left and right child nodes
 * -feature and threshold values need to be valid and consistent throughout the tree
 * -predicted label at each leaf node must be determined correctly based on the training data
 * 
 * Correspondence (the abstraction function):
 * - BinaryTree<TreeNode> root represents the decision tree, where each TreeNode corresponds
 *   to a decision node or a leaf node in the tree
 * -feature, threshold, leftChild, rightChild, and predictedLabel fields of TreeNode
 *   correspond to the attributes of a decision tree node, defining its splitting criteria
 *   and predicted label
 */


public class DecisionTreeOnBinaryTree<T, U> extends DecisionTreeClassifierSecondary<double[][], Integer> {

    private BinaryTree<TreeNode> root;

    public DecisionTreeOnBinaryTree() {
        this.root = new BinaryTree1<>();
    }

    static class TreeNode {
        String feature;
        double threshold;
        TreeNode leftChild;
        TreeNode rightChild;
        int predictedLabel;

        TreeNode(String feature, double threshold, TreeNode leftChild, TreeNode rightChild, int predictedLabel) {
            this.feature = feature;
            this.threshold = threshold;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
            this.predictedLabel = predictedLabel;
        }
    }

    //kernel methods
    @Override
    public void fit(double[][] X, int[] y) {
        this.root.clear(); //clear any existing tree
        TreeNode rootNode = buildTree(X, y, 0); //build tree recursively
        this.root.assemble(rootNode, root, root);
    }

    //recursive build tree method
    @Override
    protected TreeNode buildTree(double[][] X, int[] y, int depthLimit) {
        TreeNode root = new TreeNode(null, 0, null, null, findMajorityLabel(y));
        root.predictedLabel = findMajorityLabel(y);

        if (depthLimit <= 0 || X.length == 0 || y.length == 0) {
            return root; //if max depth is reached, return leaf node
        }

        //splitting critera
        /*
        *
        * I was hoping to implement actual AI for this whole method, but the component
        * was getting very complex at that point and so instead I opted to make a sort of decision tree simulation
        *
        */
        String feature = "feature1";
        double threshold = 2.5;

        double[][] leftX = new double[0][0];
        double[][] rightX = new double[0][0];
        int[] leftY = new int[0];
        int[] rightY = new int[0];

        //split dataset based on the hardcoded criteria
        for (int i = 0; i < X.length; i++) {
            if (X[i][0] <= threshold) {
                leftX = addRow(leftX, X[i]);
                leftY = addElement(leftY, y[i]);
            } else {
                rightX = addRow(rightX, X[i]);
                rightY = addElement(rightY, y[i]);
            }
        }

        //create children + recursively build the tree
        root.feature = feature;
        root.threshold = threshold;
        root.leftChild = buildTree(leftX, leftY, depthLimit - 1);
        root.rightChild = buildTree(rightX, rightY, depthLimit - 1);

        return root;
    }

    private static int findMajorityLabel(int[] labels) {
        //get majority label
        //I decided to use just the most common label here
        int majorityLabel = 0;
        int maxCount = 0;
        for (int label : labels) {
            int count = countOccurrences(labels, label);
            if (count > maxCount) {
                maxCount = count;
                majorityLabel = label;
            }
        }
        return majorityLabel;
    }

    private static int countOccurrences(int[] array, int value) {
        int count = 0;
        for (int num : array) {
            if (num == value) {
                count++;
            }
        }
        return count;
    }

    private static double[][] addRow(double[][] array, double[] row) {
        double[][] newArray = new double[array.length + 1][row.length];
        System.arraycopy(array, 0, newArray, 0, array.length);
        newArray[array.length] = row;
        return newArray;
    }

    private static int[] addElement(int[] array, int element) {
        int[] newArray = new int[array.length + 1];
        System.arraycopy(array, 0, newArray, 0, array.length);
        newArray[array.length] = element;
        return newArray;
    }

    @Override
    public Integer[] predict(double[][] X) {
        //predict based on the "decision tree" created from the training data
        //go through tree based on predictions
        //I also decided to limit any output types/ inputs to ints and doubles
        int size = X.length;
        Integer[] predictions = new Integer[size];
        for (int i = 0; i < size; i++) {
            predictions[i] = predictSingleNode((DecisionTreeOnBinaryTree.TreeNode) root, X[i]); //prediction has to be made for each input instance
        }
        return predictions;
    }

    protected Integer predictSingleNode(TreeNode node, double[] instance) {
        //predict the label for one instance
        if (node.leftChild == null || node.rightChild == null) {
            return node.predictedLabel; //if a leaf node is reached then a predicted label gets returned
        }
        if (instance[0] <= node.threshold) {
            return predictSingleNode(node.leftChild, instance); //traverse left sub-tree
        } else {
            return predictSingleNode(node.rightChild, instance); //traverse right sub-tree
        }
    }

    public void clear() {
        this.root.clear();
    }
    
    public DecisionTreeOnBinaryTree<T, U> newInstance() {
        return new DecisionTreeOnBinaryTree<T, U>();
    }

    @SuppressWarnings("unchecked")
    public void transferFrom(DecisionTreeClassifierEnhanced<T, U> arg0) {
        if (arg0 instanceof DecisionTreeOnBinaryTree) {
            DecisionTreeOnBinaryTree<T, U> temp = (DecisionTreeOnBinaryTree<T, U>) arg0;
            this.root.transferFrom(temp.root);
        } else{
            throw new IllegalArgumentException("Cannot transfer from a different type");
        }
    }
}
