package com.example.view;

import com.example.core.AlgorithmType;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

public class Menu extends Pane {

    private final GraphView graphView;
    private final ComboBox<String> generateGraphStrategySelector;
    private final HBox nodeCountHbox;
    private final Button generateGraphButton;
    private final CheckBox showWeightsCheckBox;
    private final CheckBox showEdgeDataCheckBox;
    private final Button clearGraphButton;
    private final Button runAlgorithmButton;
    private final Label shortestPathLabel;
    private final ComboBox<String> algorithmSelector;
    private final HBox epochCountHbox;
    private final HBox antCountHbox;


    public Menu(GraphView graphView) {
        this.graphView = graphView;

        this.setStyle("-fx-padding: 10; -fx-background-color: #f0f0f0;");

        generateGraphStrategySelector = new ComboBox<>();
        generateGraphStrategySelector.getItems().addAll("star", "random n-count");
        generateGraphStrategySelector.setValue("random n-count");

        Label label = new Label("n = ");
        TextField textField = new TextField();
        textField.setMaxWidth(40);
        textField.setText("10");
        nodeCountHbox = new HBox(5, label, textField);

        Label epochLabel = new Label("epochs = ");
        TextField epochTextField = new TextField();
        epochTextField.setMaxWidth(40);
        epochTextField.setText("1");
        epochCountHbox = new HBox(5, epochLabel, epochTextField);

        Label antCountLabel = new Label("ant count ");
        TextField antCountTextField = new TextField();
        antCountTextField.setMaxWidth(40);
        antCountTextField.setText("10");
        antCountHbox = new HBox(5, antCountLabel, antCountTextField);

        generateGraphButton = new Button("Generate graph");
        showWeightsCheckBox = new CheckBox("weights");
        showEdgeDataCheckBox = new CheckBox("edge data");
        clearGraphButton = new Button("Clear graph");
        runAlgorithmButton = new Button("Run algorithm");

        algorithmSelector = new ComboBox<>();
        algorithmSelector.getItems().addAll(AlgorithmType.ANT.toString(), AlgorithmType.DIJKSTRA.toString(),
                AlgorithmType.ANNEALING.toString());
        algorithmSelector.setValue(AlgorithmType.ANT.toString());

        shortestPathLabel = new Label("Path length: ");

        VBox vbox = new VBox(10);
        vbox.getChildren().addAll(
                generateGraphStrategySelector,
                nodeCountHbox,
                generateGraphButton,
                showWeightsCheckBox,
                showEdgeDataCheckBox,
                clearGraphButton,
                algorithmSelector,
                epochCountHbox,
                antCountHbox,
                runAlgorithmButton,
                shortestPathLabel
        );

        this.getChildren().add(vbox);

        setupEventHandlers();
    }

    private void setupEventHandlers() {
        generateGraphButton.setOnAction(_ -> generateGraph());

        showWeightsCheckBox.setOnAction(_ -> {
            if (showWeightsCheckBox.isSelected()) {
                graphView.showEdgesWeights();
            } else {
                graphView.hideEdgesWeights();
            }
        });

        showEdgeDataCheckBox.setOnAction(_ -> {
            if (showEdgeDataCheckBox.isSelected()) {
                graphView.showEdgesData();
            } else {
                graphView.hideEdgesData();
            }
        });

        clearGraphButton.setOnAction(_ -> clearGraph());
        runAlgorithmButton.setOnAction(_ -> {
            try {
                runAlgorithm();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private void generateGraph() {
        int nodeCount = Integer.parseInt(((TextField) nodeCountHbox.getChildren().getLast()).getText());
        if (nodeCount >= 0) {
            graphView.generateGraph(generateGraphStrategySelector.getValue(), nodeCount);
            shortestPathLabel.setText("Path length: ");
        }
    }

    private void clearGraph() {
        graphView.clear();
    }

    private void runAlgorithm() throws InterruptedException {
        AlgorithmType algorithmType = AlgorithmType.valueOf(algorithmSelector.getValue().toUpperCase());
        graphView.runAlgorithm(
                algorithmType,
                Integer.parseInt(((TextField) epochCountHbox.getChildren().getLast()).getText()),
                Integer.parseInt(((TextField) antCountHbox.getChildren().getLast()).getText())
        );
        graphView.showShortestWay();
        String text = shortestPathLabel.getText();
        shortestPathLabel.setText(text.split(":")[0] + ": " + graphView.getWayLength());
    }
}
