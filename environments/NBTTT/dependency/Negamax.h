

double negamax(int boards[10][10], int actions[81], int m, int curr, int depth, int branch, int player, double alpha,
               double beta, int MAX);

int alphabeta(int boards[100], int actions[81], int m, int curr, int depth, int branch, int player);