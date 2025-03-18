package com.example.core.models.graphs;

import com.example.core.models.Edge;
import com.example.core.models.Node;
import javafx.util.Pair;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;


public interface Graph extends Serializable {

    void clear();

    Set<Edge> getEdges();

    Map<Pair<Integer, Integer>, Edge> getPairEdgeMap();

    Map<Integer, Node> getNodes();

    void addNode(Node node);

    int addNode();

    boolean isEdgeExist(int sourceId, int destinationId);

    void addEdge(Edge edge);

    Set<Edge> getEdgesByNodeId(int nodeId);

    void deleteNode(int id);

    void setCountOfNodes(int countOfNodes);

    Edge getEdgeByPair(Pair<Integer, Integer> idsPair);

    int getNodesCount();
}
