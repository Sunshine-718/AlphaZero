# ======= 配置区：所有重要参数在这里集中定义 =======
import os
from PyQt5.QtGui import QColor


ENV_NAME = 'Connect4'
NETWORK_DEFAULT = 'CNN'
MODEL_NAME = 'AZ2'
MODEL_TYPE_DEFAULT = 'current'
N_PLAYOUT_DEFAULT = 500
N_PLAYOUT_MIN = 1
N_PLAYOUT_MAX = 10000

ANIMATION_INTERVAL_DEFAULT = 40   # ms
ANIMATION_INTERVAL_MIN = 10
ANIMATION_INTERVAL_MAX = 1000

PLAYER_COLOR_DEFAULT = 1         # 1: Human(X), -1: AI(X)
CELL_SIZE = 60
MARGIN = 40

CONTROL_PANEL_X = 500
CONTROL_PANEL_Y = 20
CONTROL_PANEL_WIDTH = 200
CONTROL_PANEL_HEIGHT = 460  # ↑ 高度 +60 以容纳新控件

WINDOW_TITLE = "AlphaZero Connect4 GUI"
PARAMS_PATH_FMT = './params/{model_name}_{env_name}_{network}_{model_type}.pt'

# 颜色定义（1 为红，-1 为黄）
COLOR_MAP = {
    1: QColor(255, 0, 0),      # 红色
    -1: QColor(255, 255, 0)    # 黄色
}
# ================================================

import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSpinBox, QComboBox, QPushButton, QCheckBox
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QTimer
import torch
from environments import load
from policy_value_net import PolicyValueNet
from player import Human, AlphaZeroPlayer
from game import Game

class Connect4GUI(QWidget):
    def __init__(self):
        super().__init__()
        # === 参数引用区 ===
        self.env_name = ENV_NAME
        self.network = NETWORK_DEFAULT
        self.model_name = MODEL_NAME
        self.model_type = MODEL_TYPE_DEFAULT
        self.n_playout = N_PLAYOUT_DEFAULT
        self.animation_interval = ANIMATION_INTERVAL_DEFAULT
        self.player_color = PLAYER_COLOR_DEFAULT
        self.last_probs = None
        self.last_move = None
        self.recommend_col = None
        self.show_probs = True
        self.ai_thinking_time = -1
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_drop)
        self.reload_timer = QTimer()
        self.reload_timer.setSingleShot(True)
        self.reload_timer.timeout.connect(self.auto_reload_model)

        self.env_module = load(self.env_name)
        self.env = self.env_module.Env()
        self.game = Game(self.env)

        # === 控制面板 ===
        self.control_panel = QWidget(self)
        self.layout = QVBoxLayout(self.control_panel)

        # 顶部：胜负信息标签
        self.result_label = QLabel("")
        self.layout.insertWidget(0, self.result_label)

        # n_playout 控件
        self.n_playout_label = QLabel("模拟次数 (n_playout):")
        self.n_playout_input = QSpinBox()
        self.n_playout_input.setMinimum(N_PLAYOUT_MIN)
        self.n_playout_input.setMaximum(N_PLAYOUT_MAX)
        self.n_playout_input.setValue(self.n_playout)
        self.n_playout_input.valueChanged.connect(lambda _: self.reload_timer.start(500))
        self.layout.addWidget(self.n_playout_label)
        self.layout.addWidget(self.n_playout_input)

        # 网络结构选择
        self.network_label = QLabel("网络结构:")
        self.network_choice = QComboBox()
        self.network_choice.addItems(["CNN", "ViT"])
        self.network_choice.setCurrentText(self.network)
        self.network_choice.currentIndexChanged.connect(lambda _: self.reload_timer.start(500))
        self.layout.addWidget(self.network_label)
        self.layout.addWidget(self.network_choice)

        # 模型类型选择
        self.model_type_label = QLabel("模型类型:")
        self.model_type_choice = QComboBox()
        self.model_type_choice.addItems(["current", "best"])
        self.model_type_choice.setCurrentText(self.model_type)
        self.model_type_choice.currentIndexChanged.connect(lambda _: self.reload_timer.start(500))
        self.layout.addWidget(self.model_type_label)
        self.layout.addWidget(self.model_type_choice)

        # 玩家先手选择
        self.player_label = QLabel("选择玩家先手:")
        self.player_choice = QComboBox()
        self.player_choice.addItems(["Human (X)", "AI (X)"])
        self.player_choice.currentIndexChanged.connect(lambda: (self.auto_reload_model(), self.start_game()))
        self.layout.addWidget(self.player_label)
        self.layout.addWidget(self.player_choice)

        # 动画速度
        self.speed_label = QLabel("动画速度 (ms):")
        self.speed_input = QSpinBox()
        self.speed_input.setMinimum(ANIMATION_INTERVAL_MIN)
        self.speed_input.setMaximum(ANIMATION_INTERVAL_MAX)
        self.speed_input.setValue(self.animation_interval)
        self.speed_input.valueChanged.connect(lambda _: self.reload_timer.start(500))
        self.layout.addWidget(self.speed_label)
        self.layout.addWidget(self.speed_input)

        # 是否显示推荐落子
        self.show_probs_checkbox = QCheckBox("显示推荐落子")
        self.show_probs_checkbox.setChecked(self.show_probs)
        self.show_probs_checkbox.stateChanged.connect(self.toggle_show_probs)
        self.layout.addWidget(self.show_probs_checkbox)

        # 重新开始按钮
        self.reset_button = QPushButton("重新开始")
        self.reset_button.clicked.connect(lambda: (self.auto_reload_model(), self.start_game()))
        self.layout.addWidget(self.reset_button)

        # AI 思考时间
        self.thinking_time_label = QLabel("AI 思考时间: -")
        self.layout.addWidget(self.thinking_time_label)

        # 布局与窗口设置
        self.control_panel.setGeometry(
            CONTROL_PANEL_X, CONTROL_PANEL_Y,
            CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT
        )
        self.cell_size = CELL_SIZE
        self.margin = MARGIN
        self.setFixedSize(
            self.cell_size * 7 + self.margin * 2 + CONTROL_PANEL_WIDTH,
            self.cell_size * 6 + self.margin * 2 + 60
        )
        self.setWindowTitle(WINDOW_TITLE)

        # 状态变量
        self.animating = False
        self.animation_row = -1
        self.animation_col = -1
        self.animation_color = None

        # 初始化 AlphaZero / Human 玩家
        self.auto_reload_model()
        self.current_player = [None, self.human, self.az_player]

        # 开局
        self.start_game()

    def auto_reload_model(self):
        """读取控件当前值，重新加载模型 / Player 对象"""
        self.n_playout = self.n_playout_input.value()
        self.network = self.network_choice.currentText()
        self.model_type = self.model_type_choice.currentText()
        self.animation_interval = self.speed_input.value()
        self.timer.setInterval(self.animation_interval)
        self.player_color = 1 if self.player_choice.currentText() == "Human (X)" else -1

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net_class = getattr(self.env_module, self.network)
        net = net_class(lr=0, device=device)  # lr=0 → 推理模式

        model_path = PARAMS_PATH_FMT.format(
            model_name=self.model_name,
            env_name=self.env_name,
            network=self.network,
            model_type=self.model_type
        )
        self.policy_net = PolicyValueNet(net, self.env_module.training_config['discount'], model_path)

        self.az_player = AlphaZeroPlayer(
            self.policy_net,
            c_puct=self.env_module.training_config['c_puct'],
            n_playout=self.n_playout,
            is_selfplay=0,
        )
        self.human = Human(self.policy_net)

    def start_game(self):
        self.env.reset()
        self.last_probs = None
        self.last_move = None
        self.ai_thinking_time = -1
        self.thinking_time_label.setText("AI 思考时间: -")
        self.result_label.setText("")
        self.update_recommendation()
        self.update()
        if self.env.turn == -1 * self.player_color:
            self.ai_move()

    def toggle_show_probs(self, state):
        self.show_probs = bool(state)
        self.update()

    def update_recommendation(self):
        if not self.show_probs:
            self.recommend_col = None
            return
        if self.env.turn == self.player_color and not self.env.done() and self.policy_net is not None:
            try:
                self.az_player.reset_player()
                actions, vistis = self.az_player.mcts.get_action_visits(self.env, discount=self.env_module.training_config['discount'])
                self.az_player.reset_player()
                self.recommend_col = max(actions, key=lambda x: vistis[x])
            except:
                self.recommend_col = None
        else:
            self.recommend_col = None

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_board(qp)
        qp.end()

    def draw_board(self, qp):
        state = self.env.current_state()
        board = (state[0, 0] - state[0, 1]).astype(int)
        qp.setPen(Qt.black)
        spacing = 2
        left = self.margin
        top = self.margin
        right = self.margin + self.cell_size * 7
        bottom = self.margin + self.cell_size * 6
        for c in range(1, 7):
            x = self.margin + c * self.cell_size
            qp.drawLine(x, top + spacing, x, bottom - spacing)
        for r in range(1, 6):
            y = self.margin + r * self.cell_size
            qp.drawLine(left + spacing, y, right - spacing, y)
        qp.drawLine(left, top, left, bottom)
        qp.drawLine(right, top, right, bottom)
        qp.drawLine(left, top, right, top)
        qp.drawLine(left, bottom, right, bottom)
        qp.drawRect(left + spacing, top + spacing, right - left - 2 * spacing, bottom - top - 2 * spacing)
        padding = 15
        diameter = self.cell_size - padding
        for r in range(6):
            for c in range(7):
                center_x = self.margin + c * self.cell_size + self.cell_size // 2
                center_y = self.margin + r * self.cell_size + self.cell_size // 2
                if board[r][c] == 1:
                    qp.setBrush(COLOR_MAP[1])
                elif board[r][c] == -1:
                    qp.setBrush(COLOR_MAP[-1])
                else:
                    qp.setBrush(QColor(255, 255, 255))
                qp.drawEllipse(center_x - diameter // 2, center_y - diameter // 2, diameter, diameter)
        if self.last_move:
            r, c = self.last_move
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            qp.setPen(QColor(255, 0, 0))
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(x + 5, y + 5, self.cell_size - 10, self.cell_size - 10)
        if self.show_probs and self.recommend_col is not None:
            x = self.margin + self.recommend_col * self.cell_size + self.cell_size // 2
            y = self.margin + self.cell_size * 6 + 25
            qp.setBrush(QColor(0, 200, 0))
            qp.drawEllipse(x - 6, y - 6, 12, 12)
        if self.animating and self.animation_row >= 0:
            center_x = self.margin + self.animation_col * self.cell_size + self.cell_size // 2
            center_y = self.margin + self.animation_row * self.cell_size + self.cell_size // 2
            qp.setBrush(self.animation_color)
            qp.drawEllipse(center_x - diameter // 2, center_y - diameter // 2, diameter, diameter)

    def mousePressEvent(self, event):
        """仅在棋盘区域接收点击；控制面板或空白处点击无效。"""
        # 动画或对局结束时忽略
        if self.env.done() or self.animating:
            return

        # 计算棋盘的像素边界
        x, y = event.x(), event.y()
        board_left   = self.margin
        board_top    = self.margin
        board_right  = board_left + self.cell_size * 7
        board_bottom = board_top  + self.cell_size * 6

        # 若点击不在棋盘矩形内，直接忽略
        if not (board_left <= x < board_right and board_top <= y < board_bottom):
            return

        # 以下保持原有落子逻辑
        if self.env.turn == self.player_color:
            col = (x - self.margin) // self.cell_size
            if col in self.env.valid_move():
                row = self.find_drop_row(col)
                if row != -1:
                    self.last_move = (row, col)
                    self.start_animation(
                        row, col,
                        COLOR_MAP[self.player_color],
                        lambda: self.after_move(col)
                    )

    def ai_move(self):
        if self.animating:
            return
        start = time.time()
        action, probs = self.az_player.get_action(self.env)
        print(self.az_player.mcts.cache.hit_rate())
        self.ai_thinking_time = time.time() - start
        self.thinking_time_label.setText(f"AI 思考时间: {self.ai_thinking_time:.2f} 秒")
        self.last_probs = probs if self.show_probs else None
        row = self.find_drop_row(action)
        if row != -1:
            self.last_move = (row, action)
            self.start_animation(row, action, COLOR_MAP[-self.player_color], lambda: self.after_move(action))

    def after_move(self, col):
        self.env.step(col)
        self.update_recommendation()
        self.update()
        if self.env.done():
            self.show_result()

    def find_drop_row(self, col):
        state = self.env.current_state()
        board = (state[0, 0] - state[0, 1]).astype(int)
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                return row
        return -1

    def start_animation(self, target_row, col, color, callback):
        self.animating = True
        self.animation_row = -1
        self.animation_col = col
        self.animation_color = color
        self.animation_callback = callback
        self.animation_target_row = target_row
        self.timer.start(self.animation_interval)

    def animate_drop(self):
        if self.animation_row < self.animation_target_row:
            self.animation_row += 1
            self.update()
        else:
            self.timer.stop()
            self.animating = False
            self.animation_callback()
            if self.env.turn == -1 * self.player_color and not self.env.done():
                self.ai_move()

    # 只在顶部显示胜负，不再弹窗
    def show_result(self):
        winner = self.env.winPlayer()
        if winner == self.player_color:
            self.result_label.setText("你赢了！点击重新开始。")
        elif winner == -self.player_color:
            self.result_label.setText("AlphaZero 赢了！点击重新开始。")
        else:
            self.result_label.setText("平局！点击重新开始。")

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    gui = Connect4GUI()
    gui.show()
    app.exec_()