#include "heuristic.h"
#include "utils.h"
#include "common.h"

double evalBoard(int boards[10][10], int constant, int player) {
    // evaluate all the boards and take the average value.
    double score = 0.0;
    for (int i = 1; i < 10; i++) {
        Matrix board = get_board(boards, i, player);
        int b[3][3] = {{board.a, board.b, board.c},
                       {board.d, board.e, board.f},
                       {board.g, board.h, board.i}};
        score += evalBoard_single(b, constant);
    }
    return score / 9;
}

double action_heuristic(int boards[10][10], int curr, int action, int player) {
    // Delta heuristic value
    double before = evalBoard(boards, 2, player);
    board_step(boards, curr, action, player);
    double after = evalBoard(boards, 2, player);
    board_backward(boards, curr, action);
    double score = after - before;
    return score;
}

double evalBoard_single(int boards[3][3], int constant) {
    int temp[9] = {boards[0][0], boards[0][1], boards[0][2],
                   boards[1][0], boards[1][1], boards[1][2],
                   boards[2][0], boards[2][1], boards[2][2]};
    int win = winPlayer(temp);
    if (win == 0)
        return 1000;
    else if (win == 1)
        return -1000;

    double score = 0.0;
// rotate 90 degree for 4 times (0, -90, -180, -270), because the board is symmetry
    for (int k = 0; k < 4; k++) {
        int rotated[3][3] = {{0}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                rotated[j][2 - i] = boards[i][j];
        int diag1[3] = {rotated[0][0], rotated[1][1], rotated[2][2]};
        int row1[3] = {rotated[0][0], rotated[0][1], rotated[0][2]};
        int col1[3] = {rotated[0][0], rotated[1][0], rotated[2][0]};
        if (rotated[0][0] != 0) {
// each of the value can be -1 (opponent), 0 (empty), 1 (myself).
            if (!in_(row1, -rotated[0][0])) { // Only X or only O exists in the first row
                if (count_num(rotated[0], rotated[0][0]) == 1)
// [*] if rotated[0, 0] = 1, score += 1 * 0.5, else if rotated[0, 0] = -1, score -= -1 * 0.5
                    score += (double) rotated[0][0] * 0.5;
                else
// All lines below like "score += rotated[i, j] * number" has the same rule as comment [*] above
                    score += (double) rotated[0][0];
            }
            if (!in_(col1, -rotated[0][0])) { // Only X or only O exists in the first column
                if (count_num(col1, rotated[0][0]) == 1) // number of X or O is 1
                    score += (double) rotated[0][0] * 0.5;
                else // number of X or O > 1
                    score += (double) rotated[0][0];
            }
            if (!in_(diag1, -rotated[0][0])) // Only X or only O exists in the main diagnoal,
                // do not need to take secondary diagnoal into account because the board will rotate.
                score += (double) rotated[0][0] * 0.5;
            if (rotated[0][0] == rotated[0][1] && rotated[0][0] != -rotated[0][2])
                //Only X or only O exists in the first row and two pieces are connected
                score += (double) rotated[0][0] * constant;
            if (rotated[0][0] == rotated[1][1] && rotated[0][0] != -rotated[2][2])
                //Only X or only O exists in the diagnoal and two pieces are connected
                score += (double) rotated[0][0] * constant;
            if (rotated[0][0] == rotated[1][0] && rotated[0][0] != -rotated[2][0])
                //Only X or only O exists in the first column and two pieces are connected
                score += (double) rotated[0][0] * constant;

        }
        if (rotated[0][0] == rotated[0][2] && rotated[0][0] != -rotated[0][1])
            if (rotated[1][1] == 0) {
                if (rotated[0][0] != -rotated[2][0] && rotated[0][0] != -rotated[2][2])
                    score += rotated[0][0] * constant;
                    // detecting the shape like
                    // X _ X    O _ O
                    // _ _ _ or _ _ _
                    // X _ X    X _ O
                else if ((rotated[0][0] != -rotated[2][0]) || (rotated[0][0] != -rotated[2][2]))
                    score += rotated[0][0] * constant;
                    // detecting the shape like
                    // X _ X    O _ O
                    // _ _ _ or _ _ _
                    // _ _ X    _ _ O
                else
                    score += rotated[0][0] * (constant - 2);
                    // detect the shape like
                    // X _ X    O _ O
                    // _ _ _ or _ _ _
                    // _ _ _    _ _ _
            }
        if (rotated[0][1] != 0) {
            int col2[3] = {rotated[0][1], rotated[1][1], rotated[2][1]};
            if (!in_(col2, -rotated[0][1])) { // second coloum
                if (count_num(col2, rotated[0][1]) == 1)
                    score += (double) rotated[0][1] * 0.5;
                else
                    score += (double) rotated[0][1];
            }
            if (!in_(row1, -rotated[0][1])) { // first row
                if (count_num(row1, rotated[0][1]) == 1)
                    score += (double) rotated[0][1] * 0.5;
                else
                    score += (double) rotated[0][1];
            }
            if (rotated[1][1] == rotated[0][1] && rotated[0][1] != -rotated[2][1]) // second coloum has connected pieces
                score += (double) rotated[1][1] * constant;
        }
        for (int r = 0; r < 3; r++) // update the rotated matrix
            for (int c = 0; c < 3; c++)
                boards[r][c] = rotated[r][c];
    }
    int diag11[3] = {boards[0][0], boards[1][1], boards[2][2]};
    int diag2[3] = {boards[0][2], boards[1][1], boards[2][0]};
    int row2[3] = {boards[1][0], boards[1][1], boards[1][2]};
    int col22[3] = {boards[0][1], boards[1][1], boards[2][1]};
    if (boards[1][1] != 0) {
        if (!in_(col22, -boards[1][1])) // second coloum
            score += boards[1][1];
        if (!in_(row2, -boards[1][1])) // second row
            score += boards[1][1];
        if (!in_(diag11, -boards[1][1])) // main diagnoal
            score += boards[1][1];
        if (!in_(diag2, -boards[1][1])) // secondary diagnoal
            score += boards[1][1];
    } else {
        //two disonnnected pieces
        if (boards[1][0] == boards[1][2])
            score += boards[1][0] * (constant - 1);
        if (boards[0][1] == boards[2][1])
            score += boards[0][1] * (constant - 1);
        if (boards[0][0] == boards[2][2])
            score += boards[0][0] * (constant - 0.5);
        if (boards[0][2] == boards[2][0])
            score += boards[0][2] * (constant - 0.5);
    }
    return score;
}
