package com.example.view;

import com.example.core.models.Edge;
import javafx.scene.shape.Line;
import javafx.util.Pair;

import java.util.Map;

public record EdgesData(
        Map<Pair<Integer, Integer>, Edge> edgesMap,
        Map<Pair<Integer, Integer>, Line> edgesViewsMap
) { }
