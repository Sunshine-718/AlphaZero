
typedef struct {
    int depth;
    int branch;
} Params;

typedef struct {
    int a, b, c, d, e, f, g, h, i;
} Matrix;

void board_step(int boards[10][10], int curr, int action, int player);

void board_backward(int boards[10][10], int curr, int action);

Matrix rot90(Matrix);

int winPlayer(int *board);

int in_(int arr[3], int num);

Matrix get_board(int boards[10][10], int curr, int player);

int count_num(int arr[3], int num);

int player_win(int boards[10][10]);

int is_full(int boards[10][10], int curr);

int count_pieces(int boards[10][10]);

void valid_movement(int boards[10][10], int curr, int action[9]);

double max(double a, double b);

Params dynamic_params(int boards[10][10], int depth, int branch);

void print_array(int *array, int len);

int **newMatrix(int *arr, int row, int col);

int **zeros(int row, int col);

void freeMatrix(int **mat, int row);

void print_array(int *arr, int size);

void print_2d_array(int **arr, int row, int col);

void print_board(int board[10][10]);

void reset_boards(int boards[10][10]);

int **place(int **boards, int curr, int action, int player);
