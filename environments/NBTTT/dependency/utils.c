#include "common.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* All helper function are contained in this file */

char sb[5] =  "XO.xo";


void board_step(int boards[10][10], int curr, int action, int player) {
    if (boards[curr][action] == 2)
        boards[curr][action] = player;
}

void board_backward(int boards[10][10], int curr, int action) {
    boards[curr][action] = 2;
}

int in_(int arr[3], int num) {
    if (num == arr[0] || num == arr[1] || num == arr[2])
        return 1;
    return 0;
}

Matrix get_board(int boards[10][10], int curr, int player) {
    int temp[9];
    for (int i = 1; i < 10; i++) {
        if (boards[curr][i] == 2)
            temp[i - 1] = 0;
        else
            temp[i - 1] = boards[curr][i] == player ? 1 : -1;
    }
    Matrix board = {temp[0], temp[1], temp[2],
                    temp[3], temp[4], temp[5],
                    temp[6], temp[7], temp[8]};
    return board;
}

int count_num(int arr[3], int num) {
    int c = 0;
    for (int i = 0; i < 3; i++)
        if (num == arr[i])
            c++;
    return c;
}

int winPlayer(int *board) {
    for (int i = 0; i < 3; i++) {
        if (board[i * 3] == board[i * 3 + 1] && board[i * 3] == board[i * 3 + 2]) {
            if (board[i * 3] == 0)
                continue;
            return board[i * 3] == 1 ? 0 : 1;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (board[i] == board[i + 3] && board[i] == board[i + 6]) {
            if (board[i] == 0)
                continue;
            return board[i] == 1 ? 0 : 1;
        }
    }
    if ((board[0] == board[4] && board[0] == board[8]) || (board[2] == board[4] && board[2] == board[6])) {
        if (board[4] == 0)
            return -1;
        return board[4] == 1 ? 0 : 1;
    }
    return -1;
}

int player_win(int boards[10][10]) {
    for (int r = 1; r < 10; r++) {
        int board[9] = {0};
        for (int c = 1; c < 10; c++) {
            if (boards[r][c] == 0)
                board[c - 1] = 1;
            else if (boards[r][c] == 1)
                board[c - 1] = -1;
        }
        int win = winPlayer(board);
        if (win != -1)
            return win;
    }
    return -1;
}

int is_full(int boards[10][10], int curr) {
    int count = 0;
    for (int c = 1; c < 10; c++) {
        if (boards[curr][c] != 2)
            count++;
        if (count == 9)
            return 1;
    }
    return 0;
}

int count_pieces(int boards[10][10]) {
    int count = 0;
    for (int i = 1; i < 10; i++)
        for (int j = 1; j < 10; j++)
            if (boards[i][j] != 2)
                count++;
    return count;
}

void valid_movement(int boards[10][10], int curr, int action[9]) {
    for (int c = 1; c < 10; c++) {
        if (boards[curr][c] == 2)
            action[c - 1] = 1;
        else
            action[c - 1] = 0;
    }
}

double max(double a, double b) {
    return a > b ? a : b;
}

Params dynamic_params(int boards[10][10], int depth, int branch) {
    int step = count_pieces(boards);
    int steps[7] = {0, 10, 15, 22, 27, 31, 35};
    Params res = {depth, branch};
    if (steps[0] <= step && step < steps[1])
        return res;
    else if (steps[1] <= step && step < steps[2]) {
        res.depth = 11;
        res.branch = 9;
    } else if (steps[2] <= step && step < steps[3]) {
        res.depth = 13;
        res.branch = 9;
    } else if (steps[3] <= step && step < steps[4]) {
        res.depth = 22;
        res.branch = 9;
    } else if (steps[4] <= step && step < steps[5]) {
        res.depth = 24;
        res.branch = 9;
    } else if (steps[5] <= step && step < steps[6]) {
        res.depth = 26;
        res.branch = 9;
    } else if (step >= steps[6]) {
        res.depth = 81;
        res.branch = 9;
    }
    return res;
}


int **newMatrix(int *arr, int row, int col) {
    int **mat = malloc(row * sizeof(int *));
    assert(mat != NULL);
    for (int i = 0; i < row; i++) {
        mat[i] = malloc(col * sizeof(int));
        assert(mat[i] != NULL);
    }
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            mat[i][j] = arr[i * col + j];
    return mat;
}

int **zeros(int row, int col) {
    int **mat = malloc(row * sizeof(int *));
    assert(mat != NULL);
    for (int i = 0; i < row; i++) {
        mat[i] = malloc(col * sizeof(int));
        assert(mat[i] != NULL);
    }
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            mat[i][j] = 0;
    return mat;
}

void freeMatrix(int **mat, int row) {
    if (mat != NULL) {
        for (int i = 0; i < row; i++) {
            free(mat[i]);
        }
        free(mat);
    }
}

int *zeros_1d(int len) {
    int *array = calloc(len, sizeof(int));
    assert(array != NULL);
    return array;
}

void freeArray(int *arr){
    free(arr);
}

void print_array(int *arr, int size) {
    printf("[%d", arr[0]);
    for (int i = 1; i < size; i++) {
        printf(", %d", arr[i]);
    }
    printf("]\n");
}


void print_2d_array(int **arr, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            printf("%d ", arr[i][j]);
        putchar('\n');
    }
}

void print_board_row(
        int bd[10][10],
        int a, int b, int c,
        int i, int j, int k
)
{
    printf(" %c %c %c |",sb[bd[a][i]],sb[bd[a][j]],sb[bd[a][k]]);
    printf(" %c %c %c |",sb[bd[b][i]],sb[bd[b][j]],sb[bd[b][k]]);
    printf(" %c %c %c\n",sb[bd[c][i]],sb[bd[c][j]],sb[bd[c][k]]);
}

/*********************************************************
   Print the entire board
*/
void print_board(int board[10][10])
{
    print_board_row(board,1,2,3,1,2,3);
    print_board_row(board,1,2,3,4,5,6);
    print_board_row(board,1,2,3,7,8,9);
    printf(" ------+-------+------\n");
    print_board_row(board,4,5,6,1,2,3);
    print_board_row(board,4,5,6,4,5,6);
    print_board_row(board,4,5,6,7,8,9);
    printf(" ------+-------+------\n");
    print_board_row(board,7,8,9,1,2,3);
    print_board_row(board,7,8,9,4,5,6);
    print_board_row(board,7,8,9,7,8,9);
    printf("\n");
}

int **place(int **boards, int curr, int action, int player) {
    boards[curr][action] = player;
    return boards;
}

void reset_boards(int boards[10][10]) {
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            boards[i][j] = 2;
}
