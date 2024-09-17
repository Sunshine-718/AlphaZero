#include "Negamax.h"
#include "heuristic.h"
#include <assert.h>
#include <stdlib.h>
#include "utils.h"
#include <stdio.h>
#include <math.h>

typedef struct {
    int value;
    double weight;
} Node;

int cmp(const void *a, const void *b) {
    double temp = ((Node *) b)->weight - ((Node *) a)->weight;  // descending order
    return temp > 0 ? 1 : -1;
}

void sort_actions(int actions[9], double action_values[9]) {
    Node *nodes = (Node *) malloc(9 * sizeof(Node));
    for (int i = 0; i < 9; i++) {
        nodes[i].value = actions[i];
        nodes[i].weight = action_values[i];
    }
    qsort(nodes, 9, sizeof(Node), cmp);
    for (int i = 0; i < 9; i++)
        actions[i] = nodes[i].value;
    free(nodes);
}

double negamax(int boards[10][10], int actions[81], int m, int curr, int depth, int branch, int player, double alpha,
               double beta, int MAX) {
    double value_eval = -10000;
    if (depth == 0 || is_full(boards, curr) ||
        player_win(boards) != -1) { // depth = 0 or current board is full or sombody win
        return evalBoard(boards, 2, player) * (pow(0.99, m));
    }
    int valid[9] = {0};
    valid_movement(boards, curr, valid);
    int action_list[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double action_values[9] = {-1000.};
    for (int i = 0; i < 9; i++) {
        if (valid[i])
            action_values[i] = action_heuristic(boards, curr, i + 1,
                                                player);   // calculate the delta value for each action
    }
    sort_actions(action_list, action_values); // sort the action list by heuristic value to acclerate searching
    int count = 0;
    for (int i = 0; i < 9; i++) {
        int action = action_list[i];
        if (valid[action - 1] && (count < branch || !MAX)) {
            count++;
            board_step(boards, curr, action, player);
            double negamax_value = negamax(boards, actions, m + 1, action, depth - 1, branch, !player, -beta, -alpha,
                                           !MAX);
            value_eval = max(-negamax_value, value_eval);
            board_backward(boards, curr, action);
            if (value_eval > alpha) {
                alpha = value_eval;
                actions[m] = action;
                if (alpha >= beta) {
                    return alpha;
                }
            }
        }
    }
    return alpha;
}


int alphabeta(int boards[100], int actions[81], int m, int curr, int depth, int branch, int player) {
    int mat[10][10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            mat[i][j] = boards[i * 10 + j];
        }
    }
    if (is_full(mat, curr) || player_win(mat) != -1) {
        printf("Game is over!\n");
        return -1;
    }
    negamax(mat, actions, m, curr, depth, branch, player, -10000, 10000, 1);
    int action = actions[m];
    int valid[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    valid_movement(mat, curr, valid);
    if (valid[action - 1] != 1) {
        print_array(actions, 81);
        print_array(valid, 9);
        printf("Invalid action: %d, valid[%d] = %d\n", action, action - 1, valid[action - 1]);
    }
    assert((action >= 1 && action <= 9) && (valid[action - 1] == 1));
    return action;
}
