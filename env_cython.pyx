# cython: language_level=3
# distutils: language = c++
"""env_cython.pyx — 高性能 Connect‑Four 环境
-------------------------------------------------
· 与原 Python 版 API 完全一致，便于无缝替换。
· Cython 仅在局部使用 typed‑memory‑view 加速，
  避免在 cdef 类属性上声明 buffer type（Cython 限制）。
· 兼容 Pickle，多进程共享。编译:  python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Env:
    """6×7 Connect‑Four board. 玩家棋子: 1 / -1, 空: 0"""

    # —— 私有成员：仅存储 Python 对象，禁止 buffer type ——
    cdef np.ndarray _board          # 6×7 float32 ndarray（保持 Pickle 友好）
    cdef int        _turn           # 1 / -1 当前执子方
    cdef int        _rows
    cdef int        _cols

    # ===== Pickle 协议 =====
    def __getstate__(self):
        """返回纯 Python 对象: (棋盘副本, 当前执子方)"""
        return (np.asarray(self._board, dtype=np.float32).copy(), int(self._turn))

    def __setstate__(self, state):
        board, turn = state
        self._board = np.asarray(board, dtype=np.float32).copy()
        self._turn  = int(turn)

    # ===== 只读属性：安全暴露棋盘 =====
    property board:
        def __get__(self):
            return np.asarray(self._board, dtype=np.float32).copy()

    # ===== 基础构造 & 属性 =====
    def __cinit__(self, object board=None):
        if board is not None:
            self._board = np.asarray(board, dtype=np.float32).copy()
            self._rows = self._board.shape[0]
            self._cols = self._board.shape[1]
        else:
            self._rows = 6
            self._cols = 7
            self._board = np.zeros((self._rows, self._cols), dtype=np.float32)

        cdef int i, j, count = 0
        for i in range(self._board.shape[0]):
            for j in range(self._board.shape[1]):
                if self._board[i, j] != 0:
                    count += 1
        self._turn = 1 if count % 2 == 0 else -1

    property turn:
        def __get__(self):
            return self._turn
        def __set__(self, value):
            self._turn = value

    cpdef void reset(self):
        self._board[:, :] = 0.
        self._turn = 1

    cpdef object copy(self):
        cdef Env env_copy = Env()
        env_copy._board = np.copy(self._board)
        env_copy._turn  = self._turn
        return env_copy

    # ===== 游戏逻辑 =====
    cpdef bint check_full(self):
        return not np.any(self._board == 0)

    cpdef int check_winner(self):
        """1 → 玩家1 获胜 • -1 → 玩家‑1 获胜 • 0 → 未分胜负"""
        cdef float[:, :] board = self._board  # typed‑memory‑view
        cdef int rows = board.shape[0]
        cdef int cols = board.shape[1]
        cdef int r, c
        cdef float current

        # 横向 ─▶
        for r in range(rows):
            for c in range(cols - 3):
                current = board[r, c]
                if current != 0 and current == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                    return <int>current

        # 纵向 ▼
        for r in range(rows - 3):
            for c in range(cols):
                current = board[r, c]
                if current != 0 and current == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                    return <int>current

        # 对角 ↘
        for r in range(rows - 3):
            for c in range(cols - 3):
                current = board[r, c]
                if current != 0 and current == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                    return <int>current

        # 对角 ↗
        for r in range(3, rows):
            for c in range(cols - 3):
                current = board[r, c]
                if current != 0 and current == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]:
                    return <int>current

        return 0

    cpdef bint done(self):
        return self.check_full() or self.check_winner() != 0

    cpdef list valid_move(self):
        return [i for i in range(7) if 0 in self._board[:, i]]

    cpdef list valid_mask(self):
        return [0 in self._board[:, i] for i in range(7)]

    cpdef void switch_turn(self):
        self._turn = -1 if self._turn == 1 else 1

    cpdef bint place(self, int action):
        """尝试在第 *action* 列落子。成功返回 True，否则 False（列已满）。"""
        cdef float[:, :] board = self._board
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
        """返回网络输入格式 (1, 3, 6, 7)"""
        cdef np.ndarray[np.float32_t, ndim=4] state = np.zeros((1, 3, 6, 7), dtype=np.float32)
        state[0, 0] = self._board == 1
        state[0, 1] = self._board == -1
        state[0, 2][:, :] = 1. if self._turn == 1 else -1.
        return state

    cpdef void show(self):
        cdef float[:, :] board = self._board
        cdef int r, c
        print('=' * 20)
        for r in range(6):
            row_str = []
            for c in range(7):
                val = <int>board[r, c]
                row_str.append('_' if val == 0 else ('X' if val == 1 else 'O'))
            print(' '.join(row_str))
        print('0 1 2 3 4 5 6')
        print('=' * 20)

    cpdef int winPlayer(self):
        return self.check_winner()

    # ===== 翻转相关 =====
    cpdef Env flip(self, bint inplace=False):
        """水平翻转棋盘。inplace=True → 原地翻转。"""
        cdef Env target = self if inplace else self.copy()
        target._board = target._board[:, ::-1].copy()
        return target

    cpdef int flip_action(self, int col):
        """给定原列号，返回水平翻转后的列号（若棋盘左右对称则原样返回）。"""
        cdef float[:, :] board = self._board
        cdef int rows = board.shape[0]
        cdef int cols = board.shape[1]
        cdef int r, c
        cdef bint symmetric = True

        for r in range(rows):
            for c in range(cols // 2):
                if board[r, c] != board[r, cols - 1 - c]:
                    symmetric = False
                    break
            if not symmetric:
                break

        return col if symmetric else cols - 1 - col

    cpdef tuple random_flip(self, double p=0.5):
        """以概率 *p* 随机水平翻转，返回 (env_copy, flipped_flag)。"""
        cdef Env env_copy = self.copy()
        cdef bint flipped = False
        if np.random.rand() < p:
            env_copy._board = env_copy._board[:, ::-1].copy()
            flipped = True
        return env_copy, flipped
