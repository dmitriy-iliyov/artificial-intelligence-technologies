package com.example.core.models.graphs;


import com.example.core.models.Edge;
import com.example.core.models.Node;

public class GraphFactory {

    public static Graph generateWeightedGraph() {
        return new WeightedGraph();
    }

    public static Graph generateWeightedGraphStar() {
        Graph graph = new WeightedGraph();
        graph.addNode(new Node(1));
        graph.addNode(new Node(2));
        graph.addNode(new Node(3));
        graph.addNode(new Node(4));
        graph.addNode(new Node(5));

        graph.addEdge(new Edge(1, 2, 1, 1));
        graph.addEdge(new Edge(1, 3, 21, 1));
        graph.addEdge(new Edge(1, 4, 38, 1));
        graph.addEdge(new Edge(1, 5, 1, 1));
        graph.addEdge(new Edge(2, 3, 1, 1));
        graph.addEdge(new Edge(2, 4, 7, 1));
        graph.addEdge(new Edge(2, 5, 49, 1));
        graph.addEdge(new Edge(3, 4, 1, 1));
        graph.addEdge(new Edge(3, 5, 41, 1));
        graph.addEdge(new Edge(4, 5, 1, 1));

        return graph;
    }

    public static Graph generateWeigtedGraph(int countNodes) {
        Graph graph = new WeightedGraph();
        for(int i = 0; i < countNodes; i++) {
            graph.addNode();
        }
        for (int i = 1; i < countNodes + 1; i++) {
            for (int j = 1; j < countNodes + 1; j++) {
                if (!graph.isEdgeExist(i, j) && i != j) {
                    graph.addEdge(new Edge(i, j, (int) (50 + Math.random() * 100), 1));
                }
            }
        }
        return graph;
    }

}
