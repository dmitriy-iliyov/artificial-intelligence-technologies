package com.example.core.models.graphs;

import com.example.core.models.Edge;
import com.example.core.models.Node;
import javafx.util.Pair;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class WeightedGraph implements Graph {

    private int countOfNodes = 1;
    private final Map<Integer, Node> nodes = new HashMap<>();
    private final Set<Pair<Integer, Integer>> relatedNodes = new HashSet<>();
    private final Set<Edge> edges = new HashSet<>();
    private final Map<Pair<Integer, Integer>, Edge> pairEdgeMap = new HashMap<>();


    @Override
    public int addNode() {
        Node node = new Node(countOfNodes++);
        if (!nodes.containsKey(node.getId())) {
            nodes.put(node.getId(), node);
        }
        return node.getId();
    }

    @Override
    public void addNode(Node node) {
        if (!nodes.containsKey(node.getId())) {
            nodes.put(node.getId(), node);
        }
    }

    @Override
    public boolean isEdgeExist(int sourceId, int destinationId) {
        return relatedNodes.contains(new Pair<>(sourceId, destinationId)) ||
                relatedNodes.contains(new Pair<>(destinationId, sourceId));
    }

    @Override
    public void addEdge(Edge edge) {
        if (!isEdgeExist(edge.getSourceId(), edge.getDestinationId())) {
            boolean isSourceExist = nodes.containsKey(edge.getSourceId());
            boolean isDestinationExist = nodes.containsKey(edge.getDestinationId());

            if (!isSourceExist) {
                addNode(new Node(edge.getSourceId()));
            }
            if (!isDestinationExist) {
                addNode(new Node(edge.getDestinationId()));
            }

            Pair<Integer, Integer> pair = new Pair<>(edge.getSourceId(), edge.getDestinationId());
            relatedNodes.add(pair);
            edges.add(edge);
            pairEdgeMap.put(pair, edge);
        }
    }

    @Override
    public Set<Edge> getEdgesByNodeId(int nodeId) {
        return edges.stream()
                .filter(edge -> edge.getSourceId() == nodeId || edge.getDestinationId() == nodeId)
                .collect(Collectors.toSet());
    }

    //to do
    @Override
    public void deleteNode(int id) {
        Node nodeToRemove = nodes.remove(id);
        if (nodeToRemove != null) {
            for (Node node : nodes.values()) {
//                node.getNeighbors().remove(id);
            }
        }
    }

    @Override
    public void setCountOfNodes(int countOfNodes) {
        this.countOfNodes = countOfNodes;
    }

    @Override
    public void clear() {
        nodes.clear();
        relatedNodes.clear();
        edges.clear();
        countOfNodes = 0;
    }

    @Override
    public Edge getEdgeByPair(Pair<Integer, Integer> idsPair) {
        Edge edge = pairEdgeMap.get(idsPair);
        return edge == null
                ? pairEdgeMap.get(new Pair<>(idsPair.getValue(), idsPair.getKey()))
                : edge;
    }

    @Override
    public Set<Edge> getEdges() {
        return new HashSet<>(edges);
    }

    @Override
    public Map<Pair<Integer, Integer>, Edge> getPairEdgeMap() {
        return pairEdgeMap;
    }

    @Override
    public Map<Integer, Node> getNodes() {
        return nodes;
    }

    @Override
    public int getNodesCount() {
        return nodes.values().size();
    }

    @Override
    public String toString() {
        StringBuilder graph = new StringBuilder();
        graph
                .append(nodes)
                .append("\n")
                .append("Related nodes:\n")
                .append("\t")
                .append(relatedNodes)
                .append("\n")
                .append("Graph edges:\n");
        for (Edge edge: edges) {
            graph.append("\t").append(edge).append("\n");
        }
        graph.append("\n");
        return graph.toString();
    }
}
