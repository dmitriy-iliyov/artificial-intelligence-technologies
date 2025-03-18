package com.example.view;

import com.example.core.models.Edge;
import com.example.core.models.graphs.Graph;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.scene.shape.Shape;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.util.Pair;
import lombok.Getter;

import java.util.*;

public class NodeViewFabric {

    @Getter
    private final static int CIRCLE_RADIUS = 15;
    private final static double STROKE_WIDTH = 1.5;

    private final Pane root;
    private final Graph graph;
    private StackPane selectedNode;
    private final Map<Integer, StackPane> nodesViews;
    private final Map<Pair<Integer, Integer>, Line> edgesViews;
    private final Map<Pair<Integer, Integer>, Line> edgesDataViews;
    private Random random;

    
    public NodeViewFabric(Pane root, Graph graph, Map<Integer, StackPane> nodesViews, Map<Pair<Integer, Integer>, Line> edgesViews) {
        this.root = root;
        this.graph = graph;
        this.nodesViews = nodesViews;
        this.edgesViews = edgesViews;
        this.selectedNode = null;
        this.random = new Random();
        this.edgesDataViews = new HashMap<>();
    }
    
    public StackPane addNewNode(double x, double y, int id) {
        if(isAreaEmpty(x, y, CIRCLE_RADIUS)) {
            Circle circle = new Circle(CIRCLE_RADIUS);
            circle.setFill(Color.WHITE);
            circle.setStroke(Color.BLACK);
            circle.setStrokeWidth(STROKE_WIDTH);

            Text text = new Text(String.valueOf(id));
            text.setFont(Font.font(CIRCLE_RADIUS));
            text.setFill(Color.BLACK);

            StackPane node = new StackPane(circle, text);
            node.setPrefSize(2 * CIRCLE_RADIUS, 2 * CIRCLE_RADIUS);
            node.setLayoutX(x);
            node.setLayoutY(y);

            node.setOnMouseClicked(_ -> addNewEdgeByMouseDoubleClick(node));

            nodesViews.put(id, node);

            return node;
        }
        return null;
    }
    
    private void addNewEdgeByMouseDoubleClick(StackPane node) {
        if (selectedNode == null) {
            selectedNode = node;
        } else {

            int destinationId = Integer.parseInt(((Text) node.getChildren().getLast()).getText());
            int sourceId = Integer.parseInt(((Text) selectedNode.getChildren().getLast()).getText());

            if (checkPermit(sourceId, destinationId)) {

                Dialog<Pair<String, String>> dialog = new Dialog<>();
                dialog.setTitle("Adding edge");
                dialog.setHeaderText(null);

                ButtonType okButtonType = new ButtonType("OK", ButtonBar.ButtonData.OK_DONE);
                dialog.getDialogPane().getButtonTypes().addAll(okButtonType, ButtonType.CANCEL);

                TextField weightField = new TextField();
                weightField.setPromptText("Edge weight");

                TextField pheromoneField = new TextField();
                pheromoneField.setPromptText("Edge pheromone");

                GridPane grid = new GridPane();
                grid.setHgap(10);
                grid.setVgap(10);
                grid.add(new Label("Edge weight:"), 0, 0);
                grid.add(weightField, 1, 0);
                grid.add(new Label("Edge pheromone:"), 0, 1);
                grid.add(pheromoneField, 1, 1);

                dialog.getDialogPane().setContent(grid);

                dialog.setResultConverter(dialogButton -> {
                    if (dialogButton == okButtonType) {
                        return new Pair<>(weightField.getText(), pheromoneField.getText());
                    }
                    return null;
                });

                Optional<Pair<String, String>> result = dialog.showAndWait();
                if (result.isEmpty()) {
                    selectedNode = null;
                    return;
                }

                // need for invalid input

                int edgeWeight = Integer.parseInt(result.get().getKey());
                int pheromone = Integer.parseInt(result.get().getValue());

                graph.addEdge(new Edge(sourceId, destinationId, edgeWeight, pheromone));
                printEdge(node, selectedNode, sourceId, destinationId, Color.BLACK);
            }
            selectedNode = null;
        }
    }

    public void systemAddNewEdge(StackPane sourceNode, StackPane destinationNode, int sourceId, int destinationId, Color color) {
        if (this.checkPermit(sourceId, destinationId)) {
            printEdge(sourceNode, destinationNode, sourceId, destinationId, color);
        }
    }

    private void printEdge(StackPane sourceNode, StackPane destinationNode, int sourceId, int destinationId, Color color) {
        double x1 = sourceNode.getLayoutX() + CIRCLE_RADIUS;
        double y1 = sourceNode.getLayoutY() + CIRCLE_RADIUS;
        double x2 = destinationNode.getLayoutX() + CIRCLE_RADIUS;
        double y2 = destinationNode.getLayoutY() + CIRCLE_RADIUS;

        Line edge = new Line(x1, y1, x2, y2);

        edge.setStroke(color);
        edge.setStrokeWidth(STROKE_WIDTH);

        edgesViews.put(new Pair<>(sourceId, destinationId), edge);

        root.getChildren().add(edge);

        root.getChildren().remove(sourceNode);
        root.getChildren().remove(destinationNode);

        root.getChildren().add(sourceNode);
        root.getChildren().add(destinationNode);
    }

    public void printEdgeData(Edge edge) {
        Node sourceNode = nodesViews.get(edge.getSourceId());
        Node destinationNode = nodesViews.get(edge.getDestinationId());

        double x1 = sourceNode.getLayoutX() + CIRCLE_RADIUS;
        double y1 = sourceNode.getLayoutY() + CIRCLE_RADIUS;
        double x2 = destinationNode.getLayoutX() + CIRCLE_RADIUS;
        double y2 = destinationNode.getLayoutY() + CIRCLE_RADIUS;

        Pair<Integer, Integer> pair = new Pair<>(edge.getSourceId(), edge.getDestinationId());

        Line pheromoneView = new Line(x1, y1, x2, y2);
        pheromoneView.setId("EDGE_DATA");
        pheromoneView.setStroke(Color.GRAY);
        pheromoneView.setStrokeWidth(edge.getEdgeData());
        pheromoneView.setOpacity(0.3);

//        edgesDataViews.put(pair, pheromoneView);

        root.getChildren().add(pheromoneView);

        root.getChildren().remove(sourceNode);
        root.getChildren().remove(destinationNode);

        root.getChildren().add(sourceNode);
        root.getChildren().add(destinationNode);
    }

    public void deleteEdgeData() {
        root.getChildren().removeIf(node -> {
            if (node instanceof Line) {
                return Objects.equals(node.getId(), "EDGE_DATA");
            }
            return false;
        });
    }

    private boolean isAreaEmpty(double x, double y, double radius) {
        for (Node node : root.getChildren()) {
            if (node instanceof Shape) {
                double nodeX = node.getLayoutX();
                double nodeY = node.getLayoutY();

                if (Math.sqrt(Math.pow(nodeX - x, 2) + Math.pow(nodeY - y, 2)) <= radius) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean checkPermit(int sourceId, int destinationId) {
        return !graph.isEdgeExist(sourceId, destinationId);
    }
}
