package com.example.view;

import com.example.core.Algorithm;
import com.example.core.AlgorithmType;
import com.example.core.AntAlgorithm;
import com.example.core.models.Edge;
import com.example.core.models.EdgeDataViewer;
import com.example.core.models.Node;
import com.example.core.models.graphs.Graph;
import com.example.core.models.graphs.GraphFactory;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.text.Text;
import javafx.util.Pair;

import java.util.*;

public class GraphView implements EdgeDataViewer {

    private static Color EDGE_COLOR = Color.BLACK;
    private final Pane root;
    private Graph graph;
    private final NodeViewFabric nodeViewFabric;
    private final Map<Integer, StackPane> nodesViews = new HashMap<>();
    private final Map<Pair<Integer, Integer>, Line> edgesViews = new HashMap<>();
    private final Map<AlgorithmType, Algorithm> algorithmMap;
    private Algorithm lastRunnedAlgorithm = null;
    private List<Line> shortestWayEdgesViews = new ArrayList<>();


    public GraphView(Pane root, Graph graph) {
        this.root = root;
        this.graph = graph;
        this.nodeViewFabric = new NodeViewFabric(root, graph, nodesViews, edgesViews);
        this.algorithmMap = new HashMap<>();
        algorithmMap.put(AlgorithmType.ANT, new AntAlgorithm(4, 1, 240, 0.1f, 0.6f));
    }

    public void addNode(MouseEvent event) {
        int id = graph.addNode();
        root.getChildren().add(nodeViewFabric.addNewNode(event.getX(), event.getY(), id));
    }

    public void generateGraph(String strategyName, int nodeCount) {
        this.clear();
        if (strategyName.equals("star")) {
            EDGE_COLOR = Color.BLACK;
            graph = GraphFactory.generateWeightedGraphStar();
        } else if (strategyName.equals("random n-count")) {
            EDGE_COLOR = Color.TRANSPARENT;
            graph = GraphFactory.generateWeigtedGraph(nodeCount);
        }
        graph.setCountOfNodes(graph.getNodes().values().size() + 1);

        this.printGraph(EDGE_COLOR);
    }

    public void runAlgorithm(AlgorithmType algorithmType, int epoch, int antsCount) throws InterruptedException {
        nodeViewFabric.deleteEdgeData();
        if (lastRunnedAlgorithm == null) {
            lastRunnedAlgorithm = algorithmMap.get(algorithmType);
        }
        lastRunnedAlgorithm.setEdgeDataViewer(this);
        lastRunnedAlgorithm.defaultSearch(graph, epoch, antsCount);
        if (!shortestWayEdgesViews.isEmpty()) {
            for (Line edge: shortestWayEdgesViews) {
                edge.setStroke(EDGE_COLOR);
            }
        }
        this.showEdgesData();
    }

    public void clear() {
        root.getChildren().clear();
        graph.clear();
        if (lastRunnedAlgorithm != null) {
            lastRunnedAlgorithm.clean();
        }
    }

    public void printGraph(Color color) {
        for (Node node: graph.getNodes().values()) {
            StackPane nodeView = nodeViewFabric.addNewNode(Math.random() * 1000, Math.random() * 800 - NodeViewFabric.getCIRCLE_RADIUS(), node.getId());
            root.getChildren().add(nodeView);
            nodesViews.put(node.getId(), nodeView);
        }
        for (Edge edge: graph.getEdges()) {
            StackPane sourceNode = nodesViews.get(edge.getSourceId());
            StackPane destinationNode = nodesViews.get(edge.getDestinationId());
            nodeViewFabric.systemAddNewEdge(
                    sourceNode,
                    destinationNode,
                    edge.getSourceId(),
                    edge.getDestinationId(),
                    color
            );
        }
    }

    public void paintEdges(Color color) {
        for (Line edge: edgesViews.values()) {
            edge.setStroke(color);
        }
    }

    public void showEdgesWeights() {
        for (Edge edge: graph.getEdges()) {
            Pair<Integer, Integer> pair = new Pair<>(edge.getSourceId(), edge.getDestinationId());
            Line edgeView = edgesViews.get(pair);
            Text text = new Text(String.valueOf(edge.getEdgeWeight()));
            double midpointX = (edgeView.getStartX() + edgeView.getEndX()) / 2;
            double midpointY = (edgeView.getStartY() + edgeView.getEndY()) / 2;
            text.setLayoutX(midpointX - 5);
            text.setLayoutY(midpointY - 5);
            root.getChildren().add(text);
        }
    }

    public void hideEdgesWeights() {
        for (Edge edge: graph.getEdges()) {
            Pair<Integer, Integer> pair = new Pair<>(edge.getSourceId(), edge.getDestinationId());
            Line edgeView = edgesViews.get(pair);
            double midpointX = (edgeView.getStartX() + edgeView.getEndX()) / 2;
            double midpointY = (edgeView.getStartY() + edgeView.getEndY()) / 2;

            root.getChildren().removeIf(node -> node instanceof Text &&
                    ((Text) node).getLayoutX() == midpointX - 5 &&
                    ((Text) node).getLayoutY() == midpointY - 5);
        }
    }

    public void showShortestWay() {
        List<Edge> shortestWayEdges = lastRunnedAlgorithm.getShortestWayEdges();
        for (Edge edge: shortestWayEdges) {
            Pair<Integer, Integer> pair = new Pair<>(edge.getSourceId(), edge.getDestinationId());
            Line edgeView = edgesViews.get(pair);
            edgeView.setStroke(Color.RED);
            shortestWayEdgesViews.add(edgeView);
        }
        System.out.println(shortestWayEdges);
//        System.out.println(graph.getEdges());
    }

    public int getWayLength() {
        return lastRunnedAlgorithm.getShortestWayLengths();
    }

    @Override
    public void printEdgesData(Set<Edge> edges) {
        for (Edge edge: edges) {
            nodeViewFabric.printEdgeData(edge);
        }
    }

    @Override
    public void deleteEdgesData() {
        nodeViewFabric.deleteEdgeData();
    }

    public void showEdgesData() {
        for (Edge edge: graph.getEdges()) {
            nodeViewFabric.printEdgeData(edge);
        }
    }

    public void hideEdgesData() {
        nodeViewFabric.deleteEdgeData();
    }
}
