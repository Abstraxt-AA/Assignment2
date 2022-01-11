#include "page_rank.h"

#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {


    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    bool converged = false;

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;

    double *new_solution = (double *) malloc(numNodes * sizeof(double));

    #pragma omp parallel for schedule(static, 1024) if(omp_get_max_threads() > 1)
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    while (!converged) {
        double no_outgoing_edges = 0.0;
        #pragma omp parallel for reduction(+:no_outgoing_edges) if(omp_get_max_threads() > 1) schedule(static, 1024)
        for (int i = 0; i < numNodes; ++i) {
            if (outgoing_size(g, i) == 0) {
                no_outgoing_edges += solution[i];
            }
        }
        no_outgoing_edges *= damping / numNodes;
        double global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff) if(omp_get_max_threads() > 1) schedule(dynamic, 256)
        for (int i = 0; i < numNodes; ++i) {
            new_solution[i] = 0.0;
            for (const Vertex *j = incoming_begin(g, i); j < incoming_end(g, i); ++j) {
                if (outgoing_size(g, *j) > 0) {
                    new_solution[i] += solution[*j] / outgoing_size(g, *j);
                }
            }
            new_solution[i] = (damping * new_solution[i]) + (1.0 - damping) / numNodes;
            new_solution[i] += no_outgoing_edges;
            global_diff += abs(new_solution[i] - solution[i]);
        }
        std::swap(new_solution, solution);
        converged = global_diff < convergence;
    }
    memcpy(new_solution, solution, numNodes * sizeof(double));

    /*
     CpE561 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

    */
}
