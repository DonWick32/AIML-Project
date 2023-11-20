import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0
ILLEGAL_REWARD = -2

LEFT_PAD = '  '
LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'O' else 2


def next_mark(mark):
    return 'X' if mark == 'O' else 'O'


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def after_action_state(state, action):
    """Execute an action and returns resulted state.

    Args:
        state (tuple): Board status + mark
        action (int): Action to run

    Returns:
        tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


def check_game_status(board, size):
    """Return game status by current board status.

    Args:
        board (list): Current board state
        size (int): Size of the board (n x n)

    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game (winner mark code).
    """
    for t in [1, 2]:
        # Check rows
        for i in range(size):
            if all(board[i*size + j] == t for j in range(size)):
                return t

        # Check columns
        for j in range(size):
            if all(board[i*size + j] == t for i in range(size)):
                return t

        # Check diagonals
        if all(board[i*size + i] == t for i in range(size)) or all(board[i*size + (size-i-1)] == t for i in range(size)):
            return t

    # Check for in-progress game
    for i in range(size * size):
        if board[i] == 0:
            # Still playing
            return -1

    # Draw game
    return 0

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size = 3, alpha = 0.02, show_number = False):
        low_bound = np.zeros((size* size,))
        high_bound = np.ones((size* size,))*2

        box_space = spaces.Box(low=low_bound, high=high_bound, dtype=np.float32)
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = box_space
        self.alpha = alpha
        self.set_start_mark('O')
        self.show_number = show_number
        self.size = size
        self.board_invertion = False
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def reset(self, **kwargs):
        # print(kwargs)
        self.board = [0] * self.size * self.size
        self.mark = self.start_mark
        self.done = False
        return self._get_obs()

    def step(self, action):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        if self.board[loc] ==0:
            
            self.board[loc] = tocode(self.mark)
            status = check_game_status(self.board, self.size)
            logging.debug("check_game_status board {} mark '{}'"
                          " status {}".format(self.board, self.mark, status))
            if status >= 0:
                self.done = True
                if status in [1, 2]:
                    # always called by self
                    reward = O_REWARD if self.mark == 'O' else X_REWARD

            # switch turn
            self.mark = next_mark(self.mark)
            
        else:
            self.done = False
            reward = ILLEGAL_REWARD
        
        if self.board_invertion:
            self.board_invertion = False
        else:
            self.board_invertion = True
            
        obs_, info_ = self._get_obs()
        return obs_, reward, self.done, self.done, info_

    def _get_obs(self):
        obs = np.asarray(self.board)
        
        if self.board_invertion:
            obs[obs == 1] = 3
            obs[obs == 2] = 1
            obs[obs == 3] = 2
            
        
        return (obs.astype(np.float32),{}) # obs, info

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)  # NOQA
            print('')
        else:
            self._show_board(logging.info)
            logging.info('')

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn):
        """Draw tictactoe board."""
        for j in range(0, self.size * self.size, self.size):
            def mark(i):
                return tomark(self.board[i]) if not self.show_number or self.board[i] != 0 else str(i+1)
            showfn(LEFT_PAD + '|'.join([mark(i) for i in range(j, j + self.size)]))
            if j < ((self.size * self.size) - self.size):
                showfn(LEFT_PAD + '-'*2*self.size)

    def show_turn(self, human, mark):
        self._show_turn(print if human else logging.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logging.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board, self.size)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn('')

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]


def set_log_level_by(verbosity):
    """Set log level by verbosity level.

    verbosity vs log level:

        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG

    Args:
        verbosity (int): Verbosity level given by CLI option.

    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level
