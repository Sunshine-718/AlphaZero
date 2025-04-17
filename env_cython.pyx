# cython: language_level=3
# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Env:
    cdef object board  # numpy ndarray as object
    cdef int _turn

    def __cinit__(self):
        self.board = np.zeros((6, 7), dtype=np.float32)
        self._turn = 1

    property turn:
        def __get__(self):
            return self._turn
        def __set__(self, value):
            self._turn = value

    cpdef void reset(self):
        self.board[:, :] = 0
        self._turn = 1

    cpdef object copy(self):
        cdef Env env_copy = Env()
        env_copy.board = np.copy(self.board)
        env_copy._turn = self._turn
        return env_copy

    cpdef bint check_full(self):
        return not np.any(self.board == 0)

    cpdef int check_winner(self):
        cdef np.float32_t[:, :] board = self.board
        cdef int rows = board.shape[0]
        cdef int cols = board.shape[1]
        cdef int r, c
        cdef float current
        for r in range(rows):
            for c in range(cols - 3):
                current = board[r, c]
                if current != 0 and current == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                    return int(current)
        for r in range(rows - 3):
            for c in range(cols):
                current = board[r, c]
                if current != 0 and current == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                    return int(current)
        for r in range(rows - 3):
            for c in range(cols - 3):
                current = board[r, c]
                if current != 0 and current == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                    return int(current)
        for r in range(3, rows):
            for c in range(cols - 3):
                current = board[r, c]
                if current != 0 and current == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]:
                    return int(current)
        return 0

    cpdef bint done(self):
        return self.check_full() or self.check_winner() != 0

    cpdef list valid_move(self):
        return [i for i in range(7) if 0 in self.board[:, i]]

    cpdef list valid_mask(self):
        return [0 in self.board[:, i] for i in range(7)]

    cpdef void switch_turn(self):
        self._turn = -1 if self._turn == 1 else 1

    cpdef bint place(self, int action):
        cdef np.float32_t[:, :] board = self.board
        cdef int row
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row, action] == 0:
                board[row, action] = self._turn
                return True
        return False

    cpdef void step(self, int action):
        if self.place(action):
            self.switch_turn()

    cpdef np.ndarray current_state(self):
        cdef np.ndarray[np.float32_t, ndim=4] state = np.zeros((1, 3, 6, 7), dtype=np.float32)
        state[0, 0] = self.board == 1
        state[0, 1] = self.board == -1
        state[0, 2][:, :] = 1 if self._turn == 1 else -1
        return state

    cpdef void show(self):
        cdef np.float32_t[:, :] board = self.board
        cdef int r, c
        print('=' * 20)
        for r in range(6):
            row_str = []
            for c in range(7):
                val = int(board[r, c])
                row_str.append('_' if val == 0 else ('X' if val == 1 else 'O'))
            print(' '.join(row_str))
        print('0 1 2 3 4 5 6')
        print('=' * 20)

    cpdef int winPlayer(self):
        return self.check_winner()
