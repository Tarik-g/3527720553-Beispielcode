import pygame
import numpy as np
import random
import math
import cProfile

class MonteCarlo:
    def __init__(self, simulations=1000): # 1000 Simulationen
        self.simulations = simulations

    def get_move(self, board, player):
        moves = self.get_close_moves(board)
        if len(moves) == 0:
            return None

        best_move = None
        best_win_rate = -float('inf')

        for move in moves:
            wins = 0
            for _ in range(self.simulations):
                wins += self._simulate(board.copy(), move, player)
            wins *= -1
            print("move="+str(move) + " wins="+str(wins))

            win_rate = wins / self.simulations
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_move = move

        return best_move

    def _simulate(self, board, move, player):
        board[move[0], move[1]] = player
        current_player = -player

        # Monte Carlo Search spielt eigentlich bis zum Ende
        #while True: # das ginge mit dieser Endlosschleife
        # stattdessen werden hier nur die nächsten 10 Spielzüge
        # betrachtet
        for i in range(15):
            winner = Gomoku.has_won_static(board, move[0], move[1])
            if winner != 0:
                return winner
            move = self.get_rand_move(board)
            board[move[0], move[1]] = current_player
            current_player *= -1
        return 0

    def get_rand_move(self, board):
        GRID_SIZE = len(board)
        positions = np.argwhere(board != 0)

        while True: 
            pos = random.choice(positions)
            # Ein Schritt in zufällige Richtung
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            new_x, new_y = pos[1] + dx, pos[0] + dy
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and board[new_y, new_x] == 0:
                return(new_y, new_x)

    # statt alle Spielzüge zu betrachten, werden hier nur die
    # betrachtet, die direkt neben einem anderen Stein liegen
    def get_close_moves(self, board):
        GRID_SIZE = len(board)
        occupied_positions = np.argwhere(board != 0)
        possible_moves = set()

        # Definiere alle möglichen Richtungen (oben, unten, links, rechts, diagonal)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for pos in occupied_positions:
            for dx, dy in directions:
                new_x, new_y = pos[1] + dx, pos[0] + dy
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and board[new_y, new_x] == 0:
                    possible_moves.add((new_y, new_x))

        if not possible_moves:  # Wenn keine möglichen Züge vorhanden sind
            return None

        return list(possible_moves)

class Gomoku:
    GRID_SIZE = 15
    CELL_SIZE = 40
    OFFSET = 20
    BOARD_SIZE = (GRID_SIZE-1) * CELL_SIZE + 2 * OFFSET
    STONE_RADIUS = CELL_SIZE // 2 - 5
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BACKGROUND = (220, 180, 140)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.BOARD_SIZE, self.BOARD_SIZE))
        pygame.display.set_caption("Gomoku")
        self.clock = pygame.time.Clock()
        self.board = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.current_player = 1

    def draw_board(self):
        self.screen.fill(self.BACKGROUND)
        for i in range(self.GRID_SIZE):
            pygame.draw.line(self.screen, self.BLACK, (i * self.CELL_SIZE + self.OFFSET, self.OFFSET), (i * self.CELL_SIZE + self.OFFSET, self.BOARD_SIZE - self.CELL_SIZE + self.OFFSET))
            pygame.draw.line(self.screen, self.BLACK, (self.OFFSET, i * self.CELL_SIZE + self.OFFSET), (self.BOARD_SIZE - self.CELL_SIZE + self.OFFSET, i * self.CELL_SIZE + self.OFFSET))

    def draw_stones(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.board[y, x] != 0:
                    color = self.BLACK if self.board[y, x] == 1 else self.WHITE
                    pygame.draw.circle(self.screen, color, (x * self.CELL_SIZE + self.OFFSET, y * self.CELL_SIZE + self.OFFSET), self.STONE_RADIUS)

    @staticmethod
    def has_won_static(board, x, y): # gibt 0=keiner gewonnen 1=schwarz gewonnen oder -1=weiß gewonnen zurück
        # Der Stein, der zuletzt gespielt wurde
        last_stone = board[y][x]
        if last_stone == 0:
            return 0  # Kein Stein an dieser Position

        # Alle vier Richtungen überprüfen: horizontal, vertikal, beide Diagonalen
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        GRID_SIZE = len(board)

        for dx, dy in directions:
            count = 1  # Zähler für die Anzahl der zusammenhängenden Steine

            # Überprüfe in einer Richtung
            for i in range(1, 5):
                new_x, new_y = x + dx * i, y + dy * i
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and board[new_y][new_x] == last_stone:
                    count += 1
                else:
                    break

            # Überprüfe in der entgegengesetzten Richtung
            for i in range(1, 5):
                new_x, new_y = x - dx * i, y - dy * i
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and board[new_y][new_x] == last_stone:
                    count += 1
                else:
                    break

            # Überprüfe, ob die Reihe lang genug ist
            if count >= 5:
                return board[y][x]
        return 0

    def run(self):
        mc = MonteCarlo()
        winner = 0
        running = True
        xx, yy = 0, 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Mensch spielt schwarz
                if event.type == pygame.MOUSEBUTTONDOWN and self.current_player == 1:  
                    x, y = event.pos
                    xx, yy = round((x - self.OFFSET) / self.CELL_SIZE), round((y - self.OFFSET) / self.CELL_SIZE)
                    if 0 <= xx < self.GRID_SIZE and 0 <= yy < self.GRID_SIZE and self.board[yy, xx] == 0:
                        self.board[yy, xx] = self.current_player
                        self.current_player *= -1
            self.draw_board()
            self.draw_stones()
            pygame.display.flip()
            if winner == 0:
                winner = Gomoku.has_won_static(self.board, xx, yy)
            # Monte Carlo spielt weiß
            if self.current_player == -1 and winner==0:  
                move = mc.get_move(self.board, self.current_player)
                if move:
                    self.board[move[0], move[1]] = self.current_player
                    yy = move[0]
                    xx = move[1]
                    self.current_player *= -1

            self.draw_board()
            self.draw_stones()
            pygame.display.flip()
            if winner == 0:
                winner = Gomoku.has_won_static(self.board, xx, yy)
            
            if winner == 1:
                print("Schwarz hat gewonnen.")
            if winner == -1:
                print("Weiß hat gewonnen.")

        pygame.quit()

if __name__ == "__main__":
    game = Gomoku()
    #cProfile.run('game.run()')
    game.run()
