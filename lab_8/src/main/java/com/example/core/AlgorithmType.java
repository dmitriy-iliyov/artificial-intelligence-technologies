package com.example.core;

public enum AlgorithmType {
    ANT, ANNEALING, DIJKSTRA;

    @Override
    public String toString() {
        String defaultString = super.toString();
        String part = defaultString.substring(1).toLowerCase();
        return defaultString.charAt(0) + part;
    }
}
