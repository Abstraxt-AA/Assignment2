#ifndef __BFS_H__
#define __BFS_H__


#include "common/graph.h"

struct solution
{
    int *distances;
};

struct vertex_set {
    int total_count;
    int *counts;
    int max_vertices_per_thread;
    int **vertices;
};

void bfs_top_down(Graph graph, solution* sol);
void bfs_bottom_up(Graph graph, solution* sol);
void bfs_hybrid(Graph graph, solution* sol);

#endif
