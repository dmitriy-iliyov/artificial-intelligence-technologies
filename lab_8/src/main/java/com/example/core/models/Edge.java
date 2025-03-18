package com.example.core.models;


import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

@Data
@AllArgsConstructor
public class Edge implements Serializable {
    private final int sourceId;
    private final int destinationId;
    private final int edgeWeight;
    private double edgeData;

    @Override
    public String toString() {
        return "\nEdge {" + "sourceId:" + sourceId +
                "; destinationId:" + destinationId +
                "; edgeWeight:" + edgeWeight +
                "; edgeData:" + edgeData + "}";
    }
}
