package com.example.core;


import com.example.core.models.Edge;
import com.example.core.models.EdgeDataViewer;
import com.example.core.models.graphs.Graph;

import java.util.List;

public interface Algorithm {
    void defaultSearch(Graph graph, int epochs, int antCount) throws InterruptedException;

    void setEdgeDataViewer(EdgeDataViewer edgeDataViewer);

    int getShortestWayLengths();

    List<Edge> getShortestWayEdges();

    void clean();
}
