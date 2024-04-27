import components.decisiontree.DecisionTreeOnBinaryTree;
import org.junit.Test;

public class DecisionTreeOnBinaryTreeTest {

    private DecisionTreeOnBinaryTree<double[][], Integer> tree;

    @Before
    public void setUp() {
        tree = new DecisionTreeOnBinaryTree<>();
    }

    @Test
    public void testFit() {
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
        int[] y = {0, 1};
        tree.fit(X, y);
        assertNotNull(tree.root);
    }

    @Test
    public void testBuildTree() {
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        int[] y = {0, 1, 0};
        //build the tree with a depth limit of 3
        DecisionTreeOnBinaryTree.TreeNode rootNode = tree.buildTree(X, y, 3);
        assertNotNull(rootNode);
    }

    @Test
    public void testPredict() {
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
        int[] y = {0, 1};
        tree.fit(X, y);
        Integer[] predictions = tree.predict(X);
        assertNotNull(predictions);
    }

    @Test
    public void testPredictSingleNode() {
        DecisionTreeOnBinaryTree.TreeNode node = new DecisionTreeOnBinaryTree.TreeNode("feature1", 2.5, null, null, 1);
        double[] instance1 = {1.0};
        assertEquals(1, (int) tree.predictSingleNode(node, instance1)); //left
        double[] instance2 = {3.0};
        assertEquals(1, (int) tree.predictSingleNode(node, instance2)); //right
    }

    @Test
    public void testClear() {
        tree.fit(new double[][]{{1.0, 2.0}}, new int[]{0});
        assertNotNull(tree.root);
        tree.clear();
        assertNull(tree.root);
    }

    @Test
    public void testNewInstance() {
        DecisionTreeOnBinaryTree<double[][], Integer> newInstance = tree.newInstance();
        assertNotNull(newInstance);
        assertTrue(newInstance instanceof DecisionTreeOnBinaryTree);
    }

    @Test
    public void testTransferFrom() {
        DecisionTreeOnBinaryTree<double[][], Integer> other = new DecisionTreeOnBinaryTree<>();
        other.fit(new double[][]{{1.0, 2.0}}, new int[]{0});
        tree.transferFrom(other);
        assertNotNull(tree.root);
        assertThrows(IllegalArgumentException.class, () -> tree.transferFrom(new DecisionTreeClassifierSecondary<>()));
    }
}
