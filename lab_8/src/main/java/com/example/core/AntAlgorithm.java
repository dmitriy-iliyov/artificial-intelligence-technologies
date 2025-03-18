package com.example.core;

import com.example.core.models.Edge;
import com.example.core.models.EdgeDataViewer;
import com.example.core.models.graphs.Graph;
import javafx.util.Pair;
import lombok.Data;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

@Data
public class AntAlgorithm implements Algorithm {

    private static final Logger LOGGER = Logger.getLogger(AntAlgorithm.class.getName());
    private final float beta;
    private final float alpha;
    private final float tau;
    private final float q;
    private final float p;
    private final Random random;
    private final List<Integer> nodeBlackList;
    private int shortestWayLengths;
    private List<Edge> shortestWayEdges;
    private EdgeDataViewer edgeDataViewer;


    public AntAlgorithm(float beta, float alpha, float q, float tau, float p) {
        this.beta = beta;
        this.alpha = alpha;
        this.q = q;
        this.tau = tau;
        this.p = p;
        this.nodeBlackList = new ArrayList<>();
        this.random = new Random();
        this.shortestWayLengths = Integer.MAX_VALUE;
        this.shortestWayEdges = new ArrayList<>();
    }

    @Override
    public void defaultSearch(Graph graph, int epochs, int antCount) {

        this.setTaus(graph);
        int nodesCount = graph.getNodes().values().size();
        int eliteAntCount = antCount/3;
        antCount += eliteAntCount;

        List<Integer> eliteAntIndex = new ArrayList<>();

        for (int i = 0; i < eliteAntCount; i++) {
            eliteAntIndex.add((int) (1 + Math.random() * (antCount + 1)));
        }

        for (int epochNumber = 0; epochNumber < epochs; epochNumber++) {

            List<Integer> waysLength = new ArrayList<>();
            List<Set<Edge>> passedEdgesSets = new ArrayList<>();

            for (int antNumber = 0; antNumber < antCount; antNumber++) {

                // set new start (nodeId) for each epoch
                int randNodeNum = random.nextInt(1, nodesCount);
                int startNodeId = (graph.getNodes().values().stream().toList()).get(randNodeNum).getId();
                int nodeId = startNodeId;

                Set<Edge> passedEdges = new HashSet<>();
                int passedWayLengths = 0;

                for (int i = 0; i < nodesCount; i++) {

                    LOGGER.info("Ant №" + i);
                    // get permit edges set map prob:edge to get edge by prob after get randProb
                    Set<Edge> nodeEdges = this.getPermittedEdge(graph.getEdgesByNodeId(nodeId));
                    Map<Float, Edge> probEdgeMap = new HashMap<>();
                    float previousProb = 0f;
                    float maxProb = Float.MAX_VALUE;
                    Edge maxProbEdge = null;

                    // make list of probabilities as number line which lengths is 1
                    for (Edge edge: nodeEdges) {
                        float currentProb = this.probabilityTransition(edge, nodeEdges);
                        if (eliteAntIndex.contains(antNumber) && maxProb > currentProb) {
                            maxProb = currentProb;
                            maxProbEdge = edge;
                        }
                        probEdgeMap.put(previousProb + currentProb, edge);
                        previousProb += currentProb;
                    }

                    Edge chosenEdge = null;
                    if (eliteAntIndex.contains(antNumber)) {
                        chosenEdge = maxProbEdge;
                    } else {
                        LOGGER.info("Current node id: " + nodeId);
                        LOGGER.info("Node edges: " + nodeEdges);
                        LOGGER.info("Probability/Edge map: \n" + probEdgeMap);

                        // find interval according to rand prob and set next node
                        float randProb = random.nextFloat();
                        Float [] probs = probEdgeMap.keySet().toArray(new Float[0]);

                        LOGGER.info("rand prob: " + randProb);
                        LOGGER.info("probs: " + Arrays.toString(probs));

                        for (float prob : probs) {
                            if (randProb <= prob) {
                                chosenEdge = probEdgeMap.get(prob);
                                break;
                            }
                        }
                    }

                    if (i == nodesCount - 1) {
                        chosenEdge = graph.getEdgeByPair(new Pair<>(nodeId, startNodeId));
                    }

                    if (chosenEdge == null) {
                        throw new IllegalStateException("chosen edge is null");
                    }

                    LOGGER.info("Chosen edge: " + chosenEdge);

                    nodeBlackList.add(nodeId);
                    nodeId = chosenEdge.getDestinationId() == nodeId ? chosenEdge.getSourceId() : chosenEdge.getDestinationId();

                    LOGGER.info("Next node id: " + nodeId);
                    LOGGER.info("Node black list:" + nodeBlackList);

                    passedWayLengths += chosenEdge.getEdgeWeight();
                    passedEdges.add(chosenEdge);
                }

                // clear blacklist
                nodeBlackList.clear();

                if (passedWayLengths < shortestWayLengths && passedWayLengths != 0) {
                    shortestWayLengths = passedWayLengths;
                    shortestWayEdges.clear();
                    shortestWayEdges.addAll(passedEdges);
                }

                // list of passed edges for next ant
                waysLength.add(passedWayLengths);
                passedEdgesSets.add(passedEdges);
            }

            float minTau = 0.0001f;
            for (Edge edge: graph.getEdges()) {
                edge.setEdgeData(Math.max(p * edge.getEdgeData(), minTau));
            }

            //update taus(pheromones)
            for (int i = 0; i < passedEdgesSets.size(); i++) {
                this.updateTaus(passedEdgesSets.get(i), waysLength.get(i));
            }
        }
        Optional<Double> maxTauOpt = graph.getEdges().stream()
                .map(Edge::getEdgeData)
                .max(Double::compare);

        if (maxTauOpt.isPresent() && maxTauOpt.get() > 0) {
            double maxTau = maxTauOpt.get();
            for (Edge edge : graph.getEdges()) {
                edge.setEdgeData((edge.getEdgeData() / maxTau) * 10);
            }
        }
    }

    private void setTaus(Graph graph) {
        for (Edge edge: graph.getEdges()) {
            edge.setEdgeData(tau);
        }
    }

    private float probabilityTransition(Edge currentEdge, Set<Edge> edges) {
        float denominator = 0F;
        for (Edge edge: edges) {
            denominator += this.calculateNumerator(edge);
        }
        return this.calculateNumerator(currentEdge)/denominator;
    }

    private float calculateNumerator(Edge edge) {
        return (float) (100 * Math.pow(edge.getEdgeData(), alpha) * Math.pow(1.0 / edge.getEdgeWeight(), beta));
    }

    private Set<Edge> getPermittedEdge(Set<Edge> nodeEdges) {
        return nodeEdges.stream()
                .filter(edge -> !nodeBlackList.contains(edge.getDestinationId()) && !nodeBlackList.contains(edge.getSourceId()))
                .collect(Collectors.toSet());
    }

    private void updateTaus(Set<Edge> passedEdges, int passedWayLengths) {
        float deltaTau = q / passedWayLengths;
//        System.out.println(deltaTau);
        for (Edge edge: passedEdges) {
            edge.setEdgeData(edge.getEdgeData() + deltaTau);
        }
    }

    @Override
    public void setEdgeDataViewer(EdgeDataViewer edgeDataViewer) {
        this.edgeDataViewer = edgeDataViewer;
    }

    @Override
    public int getShortestWayLengths() {
        return shortestWayLengths;
    }

    @Override
    public List<Edge> getShortestWayEdges() {
        return shortestWayEdges;
    }

    @Override
    public void clean() {
        this.nodeBlackList.clear();
        this.shortestWayLengths = Integer.MAX_VALUE;
        this.shortestWayEdges = new ArrayList<>();
    }
}
