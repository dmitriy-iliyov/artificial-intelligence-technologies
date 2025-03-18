package com.example;

import com.example.configs.LoggingConfig;
import com.example.core.models.graphs.Graph;
import com.example.core.models.graphs.WeightedGraph;
import com.example.view.GraphView;
import com.example.view.Menu;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;


public class GraphRunner extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {

        LoggingConfig.configureLogging();

        BorderPane root = new BorderPane();
        Pane graphRoot = new Pane();
        root.setStyle("-fx-background-color: white;");
        root.setCenter(graphRoot);
        Graph graph = new WeightedGraph();
        GraphView graphView = new GraphView(graphRoot, graph);
        Menu menu = new Menu(graphView);
        root.setLeft(menu);

        graphRoot.setOnMouseClicked(event -> {
            if (event.getClickCount() == 2) {
                graphView.addNode(event);
            }
        });

        Scene scene = new Scene(root, 1200, 800);
        primaryStage.setTitle("Graph");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}
