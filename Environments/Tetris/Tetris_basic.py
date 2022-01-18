import sys
import numpy as np
import math
from random import randint
from collections import deque
import pygame

from Environments.multiagentenv import MultiAgentEnv

HEIGHT = 22
WIDTH = 12
INTERVAL = 40
PIECE_SIZE = 24
PIECE_GRID_SIZE = PIECE_SIZE + 1


def rotate_perp(vector):
    n = int_sqrt(vector)
    array_2d = np.array(vector).reshape((n, n))
    list_of_tuples = zip(*array_2d[::-1])
    rotated = [list(elem) for elem in list_of_tuples]
    return flatten(rotated)


def flatten(lst):
    return [x for y in lst for x in y]


def int_sqrt(vector):
    return int(math.sqrt(len(vector)))


class Tetris_single(MultiAgentEnv):
    def __init__(self,
                 Count=0,
                 MinoName=None,
                 MinoData=None,
                 MinoColor=None,
                 Ghost=False,
                 Hold=False,
                 InitGravitiy=0,
                 MaxGravity=20,
                 TimeOut=0,
                 Preview=1,
                 History=False
                 ):
        self.MinoName = MinoName
        self.MinoData = dict()
        for mino, data in MinoData.items():
            input_data = [data]
            rotated_data = data
            for rotate in range(1, 4):
                rotated_data = rotate_perp(rotated_data)
                input_data.append(rotated_data)
            self.MinoData[mino] = input_data

        self.Turn = randint(0, 3)
        self.Current = randint(0, len(self.MinoName) - 1)
        self.Preview = deque()
        for index in range(Preview):
            self.Preview.append(randint(0, len(self.MinoName) - 1))
        self.Type = self.MinoData[self.MinoName[self.Current]]
        self.Color = MinoColor
        self.Data = self.Type[self.Turn]
        self.Size = int_sqrt(self.Data)
        self.Xpos = randint(2, 8 - self.Size)
        self.Ypos = 1 - self.Size

        self.Field = []

        pygame.init()
        self.SmallFont = pygame.font.SysFont(None, 36)
        self.LargeFont = pygame.font.SysFont(None, 72)
        pygame.key.set_repeat(30, 30)
        self.Screen = pygame.display.set_mode((600, 600))
        self.Clock = pygame.time.Clock()

        self.Score = 0
        self.Fire = Count + INTERVAL

    def reset(self):
        self.Score = 0
        count = 0

        self.Turn = randint(0, 3)
        self.Current = randint(0, len(self.MinoName) - 1)
        self.Type = self.MinoData[self.MinoName[self.Current]]
        self.Data = self.Type[self.Turn]
        self.Size = int_sqrt(self.Data)
        self.Xpos = randint(2, 8 - self.Size)
        self.Ypos = 1 - self.Size
        preview = len(self.Preview)
        self.Preview = deque()
        for index in range(preview):
            self.Preview.append(randint(0, len(self.MinoName) - 1))

        self.set_game_field()

    # region custom module
    def go_next_block(self):
        """ 블록을 생성하고, 다음 블록으로 전환한다 """
        self.Current = self.Preview.popleft()
        self.Preview.append(randint(0, len(self.MinoName) - 1))
        self.Turn = randint(0, 3)
        self.Type = self.MinoData[self.MinoName[self.Current]]
        self.Data = self.Type[self.Turn]
        self.Size = int_sqrt(self.Data)
        self.Xpos = randint(2, 8 - self.Size)
        self.Ypos = 1 - self.Size

    def set_game_field(self):
        """ TODO : 필드 구성을 위해 FIELD 값을 세팅한다. """
        for i in range(HEIGHT - 1):
            self.Field.insert(0, [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8])

        self.Field.insert(HEIGHT - 1, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])

    def update(self, count):
        """ 블록 상태 갱신 (소거한 단의 수를 반환한다) """
        # 아래로 충돌?
        erased = 0
        if self.is_overlapped(self.Xpos, self.Ypos + 1, self.Turn):
            for y_offset in range(self.Size):
                for x_offset in range(self.Size):
                    index = y_offset * self.Size + x_offset
                    val = self.Data[index]
                    if 0 <= self.Ypos + y_offset < HEIGHT and 0 <= self.Xpos + x_offset < WIDTH and val != 0:
                        self.Field[self.Ypos + y_offset][
                            self.Xpos + x_offset] = val  ## 값을 채우고, erase_line()을 통해 삭제되도록 한다.

            erased = self.erase_line()
            self.go_next_block()

        if self.Fire < count:
            self.Fire = count + INTERVAL
            self.Ypos += 1
        return erased

    def erase_line(self):
        """ TODO : 행이 모두 찬 단을 지운다. 그리고, 소거한 단의 수를 반환한다 """
        erased = 0
        ypos = HEIGHT - 2
        while ypos >= 0:
            if all(self.Field[ypos]):
                del self.Field[ypos]
                self.Field.insert(0, [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8])
                erased += 1
            else:
                ypos -= 1
        return erased

    def is_overlapped(self, xpos, ypos, turn):
        """ TODO : 블록이 벽이나 땅의 블록과 충돌하는지 아닌지 """
        data = self.Type[turn]
        for y_offset in range(self.Size):
            for x_offset in range(self.Size):
                index = y_offset * self.Size + x_offset
                val = data[index]

                if 0 <= xpos + x_offset < WIDTH and 0 <= ypos + y_offset < HEIGHT:
                    if val != 0 and self.Field[ypos + y_offset][xpos + x_offset] != 0:
                        return True
        return False

    def is_game_over(self):
        """ TODO : 게임 오버인지 아닌지 """
        data = self.Type[self.Turn]
        for y_offset in range(self.Size):
            for x_offset in range(self.Size):
                index = y_offset * self.Size + x_offset
                val = data[index]

                if (self.Ypos + 1) + y_offset <= 0:
                    if val != 0 and self.Field[0][self.Xpos + x_offset] != 0:
                        return True
        return False

    # endregion
    # region draw section
    def draw(self):
        """ 블록을 그린다 """
        for y_offset in range(self.Size):
            for x_offset in range(self.Size):
                index = y_offset * self.Size + x_offset
                val = self.Data[index]
                if 0 <= y_offset + self.Ypos < HEIGHT and 0 <= x_offset + self.Xpos < WIDTH and val != 0:
                    f_xpos = PIECE_GRID_SIZE + (x_offset + self.Xpos) * PIECE_GRID_SIZE
                    f_ypos = PIECE_GRID_SIZE + (y_offset + self.Ypos) * PIECE_GRID_SIZE
                    pygame.draw.rect(self.Screen, self.Color[self.MinoName[self.Current]],
                                     (f_xpos,
                                      f_ypos,
                                      PIECE_SIZE,
                                      PIECE_SIZE))

    def draw_game_field(self):
        """ TODO : 필드를 그린다 """
        for y_offset in range(HEIGHT):
            for x_offset in range(WIDTH):
                val = self.Field[y_offset][x_offset]
                color = [0, 0, 0]
                if val != 0:
                    color = [125, 125, 125]
                pygame.draw.rect(self.Screen,
                                 color,
                                 (PIECE_GRID_SIZE + x_offset * PIECE_GRID_SIZE,
                                  PIECE_GRID_SIZE + y_offset * PIECE_GRID_SIZE,
                                  PIECE_SIZE,
                                  PIECE_SIZE))

    def draw_next_block(self):
        """ TODO : 다음 블록을 그린다 """
        ## 블록의 조각(piece)의 데이터를 구한다.

        data_type = self.MinoData[self.MinoName[self.Preview[0]]]
        data = data_type[0]
        size = int_sqrt(data)
        for y_offset in range(size):
            for x_offset in range(size):
                index = y_offset * size + x_offset
                val = data[index]
                # if 0 <= y_offset + self.ypos < HEIGHT and \
                #   0 <= x_offset + self.xpos < WIDTH and
                if val != 0:  ## 이 조건은 중요함! 0까지 그림을 그린다면, 쌓인 블록이 순간적으로 검정색이 됨.
                    ## f_xpos = filed에서의 xpos를 계산함
                    f_xpos = 460 + (x_offset) * PIECE_GRID_SIZE
                    f_ypos = 100 + (y_offset) * PIECE_GRID_SIZE
                    pygame.draw.rect(self.Screen, self.Color[self.MinoName[self.Preview[0]]],
                                     (f_xpos,
                                      f_ypos,
                                      PIECE_SIZE,
                                      PIECE_SIZE))

    def draw_score(self):
        """ TODO : 점수를 표시한다. """
        score_str = str(self.Score).zfill(6)
        score_image = self.SmallFont.render(score_str, True, (0, 255, 0))
        self.Screen.blit(score_image, (500, 30))

    def draw_gameover_message(self):
        """ TODO : 'Game Over' 문구를 표시한다 """
        score_image = self.SmallFont.render("GAME OVER", True, (255, 0, 0))
        self.Screen.blit(score_image, (200, 300))

    # endregion
    # region play module
    def drop_block(self, xpos):
        ypos = 0
        for idx in range(0, 20):
            rtn = self.is_overlapped(xpos, 20 - idx, self.Turn)
            if not rtn:
                ypos = 20 - idx
                break
        return ypos

    def runGame(self, count):

        self.Clock.tick(10)
        self.Screen.fill((0, 0, 0))

        key = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                key = event.key
            elif event.type == pygame.KEYUP:
                key = None

        game_over = self.is_game_over()
        if not game_over:
            count += 5
            if count % 1000 == 0:
                interval = max(1, 40 - 2)
            erased = self.update(count)

            if erased > 0:
                self.Score += (2 ** erased) * 100

            # 키 이벤트 처리
            next_x, next_y, next_t = \
                self.Xpos, self.Ypos, self.Turn
            if key == pygame.K_UP:
                next_t = (next_t + 1) % 4
            elif key == pygame.K_RIGHT:
                next_x += 1
            elif key == pygame.K_LEFT:
                next_x -= 1
            elif key == pygame.K_DOWN:
                next_y += 1
            elif key == pygame.K_SPACE:
                next_y = self.drop_block(next_x)

            if not self.is_overlapped(next_x, next_y, next_t):
                self.Xpos = next_x
                self.Ypos = next_y
                self.Turn = next_t
                self.Data = self.Type[self.Turn]

        # 게임필드 그리기
        self.draw_game_field()
        pygame.display.update()
        # 낙하 중인 블록 그리기
        self.draw()
        pygame.display.update()
        # 다음 블록 그리기
        self.draw_next_block()
        pygame.display.update()
        # 점수 나타내기
        self.draw_score()
        pygame.display.update()
        # 게임 오버 메시지
        if game_over:
            self.draw_gameover_message()

        pygame.display.update()
        return game_over, count

    # end region
    def step(self, actions):
        count = 0
        while True:
            terminated, count = self.runGame(count)
            if terminated:
                break
        reward = 0
        return reward, terminated, {}


class Tetris_multi(MultiAgentEnv):
    def __init__(self,
                 Agent=3,
                 Count=0,
                 MinoName=None,
                 MinoData=None,
                 MinoColor=None,
                 Ghost=False,
                 Hold=False,
                 InitGravitiy=0,
                 MaxGravity=20,
                 TimeOut=0,
                 Preview=1,
                 History=False
                 ):
        self.NumAgents = Agent
        self.MinoName = MinoName
        self.Color = MinoColor
        self.MinoData = dict()
        for mino, data in MinoData.items():
            input_data = [data]
            rotated_data = data
            for rotate in range(1, 4):
                rotated_data = rotate_perp(rotated_data)
                input_data.append(rotated_data)
            self.MinoData[mino] = input_data

        self.Turn = []
        self.Current = []
        self.Preview = []
        self.Type = []
        self.Data = []
        self.Size = []
        self.Xpos = []
        self.Ypos = []
        self.Field = []
        self.Score = []
        self.Fire = []

        for agent in range(self.NumAgents):
            self.Turn.append(randint(0, 3))
            self.Current.append(randint(0, len(self.MinoName) - 1))
            self.Preview.append(deque())
            self.Type.append(self.MinoData[self.MinoName[self.Current[agent]]])
            for index in range(Preview):
                self.Preview[agent].append(randint(0, len(self.MinoName) - 1))
            self.Data.append(self.Type[agent][self.Turn[agent]])
            self.Size.append(int_sqrt(self.Data[agent]))
            self.Xpos.append(randint(2, 8 - self.Size[agent]))
            self.Ypos.append(1 - self.Size[agent])
            self.Field.append([])
            self.Score.append(0)
            self.Fire.append(Count + INTERVAL)

        pygame.init()
        self.SmallFont = pygame.font.SysFont(None, 36)
        self.LargeFont = pygame.font.SysFont(None, 72)
        pygame.key.set_repeat(30, 30)
        self.Screen = pygame.display.set_mode((600 * Agent, 600))
        self.Clock = pygame.time.Clock()

    def reset(self):
        for agent in range(self.NumAgents):
            self.Turn[agent] = randint(0, 3)
            self.Current[agent] = randint(0, len(self.MinoName) - 1)
            preview = len(self.Preview[agent])
            self.Preview[agent] = deque()
            self.Type[agent] = self.MinoData[self.MinoName[self.Current[agent]]]
            for index in range(preview):
                self.Preview[agent].append(randint(0, len(self.MinoName) - 1))
            self.Data[agent] = self.Type[agent][self.Turn[agent]]
            self.Size[agent] = int_sqrt(self.Data[agent])
            self.Xpos[agent] = randint(2, 8 - self.Size[agent])
            self.Ypos[agent] = 1 - self.Size[agent]
            self.Score[agent] = 0
            self.Fire[agent] = INTERVAL

        self.set_game_field()

    # region custom module
    def go_next_block(self, agent_idx):
        """ 블록을 생성하고, 다음 블록으로 전환한다 """
        self.Current[agent_idx] = self.Preview[agent_idx].popleft()
        self.Preview[agent_idx].append(randint(0, len(self.MinoName) - 1))
        self.Turn[agent_idx] = randint(0, 3)
        self.Type[agent_idx] = self.MinoData[self.MinoName[self.Current[agent_idx]]]
        self.Data[agent_idx] = self.Type[agent_idx][self.Turn[agent_idx]]
        self.Size[agent_idx] = int_sqrt(self.Data[agent_idx])
        self.Xpos[agent_idx] = randint(2, 8 - self.Size[agent_idx])
        self.Ypos[agent_idx] = 1 - self.Size[agent_idx]

    def set_game_field(self):
        """ TODO : 필드 구성을 위해 FIELD 값을 세팅한다. """
        for agent in range(self.NumAgents):
            for i in range(HEIGHT - 1):
                self.Field[agent].insert(0, [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8])

            self.Field[agent].insert(HEIGHT - 1, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])

    def update(self, count, agent_idx):
        """ 블록 상태 갱신 (소거한 단의 수를 반환한다) """
        # 아래로 충돌?
        erased = 0
        if self.is_overlapped(self.Xpos[agent_idx], self.Ypos[agent_idx], self.Turn[agent_idx], agent_idx):
            for y_offset in range(self.Size[agent_idx]):
                for x_offset in range(self.Size[agent_idx]):
                    index = y_offset * self.Size[agent_idx] + x_offset
                    val = self.Data[agent_idx][index]
                    if 0 <= self.Ypos[agent_idx] + y_offset < HEIGHT and\
                            0 <= self.Xpos[agent_idx] + x_offset < WIDTH and val != 0:
                        self.Field[agent_idx][self.Ypos[agent_idx] + y_offset][
                            self.Xpos[agent_idx] + x_offset] = val  ## 값을 채우고, erase_line()을 통해 삭제되도록 한다.

            erased = self.erase_line(agent_idx)
            self.go_next_block(agent_idx)

        if self.Fire[agent_idx] < count:
            self.Fire[agent_idx] = count + INTERVAL
            self.Ypos[agent_idx] += 1
        return erased

    def erase_line(self, agent_idx):
        """ TODO : 행이 모두 찬 단을 지운다. 그리고, 소거한 단의 수를 반환한다 """
        erased = 0
        ypos = HEIGHT - 2
        while ypos >= 0:
            if all(self.Field[agent_idx][ypos]):
                del self.Field[agent_idx][ypos]
                self.Field[agent_idx].insert(0, [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8])
                erased += 1
            else:
                ypos -= 1
        return erased

    def is_overlapped(self, xpos, ypos, turn, agent_idx):
        """ TODO : 블록이 벽이나 땅의 블록과 충돌하는지 아닌지 """
        data = self.Type[agent_idx][turn]
        for y_offset in range(self.Size[agent_idx]):
            for x_offset in range(self.Size[agent_idx]):
                index = y_offset * self.Size[agent_idx] + x_offset
                val = data[index]

                if 0 <= xpos + x_offset < WIDTH and 0 <= ypos + 1 + y_offset < HEIGHT:
                    if val != 0 and self.Field[agent_idx][ypos + 1 + y_offset][xpos + x_offset] != 0:
                        return True
        return False

    def is_game_over(self, agent_idx):
        """ TODO : 게임 오버인지 아닌지 """
        data = self.Type[agent_idx][self.Turn[agent_idx]]
        for y_offset in range(self.Size[agent_idx]):
            for x_offset in range(self.Size[agent_idx]):
                index = y_offset * self.Size[agent_idx] + x_offset
                val = data[index]

                if (self.Ypos[agent_idx] + 1) + y_offset <= 0:
                    if val != 0 and self.Field[agent_idx][0][self.Xpos[agent_idx] + x_offset] != 0:
                        return True
        return False

    # endregion
    # region draw section
    def draw(self):
        """ 블록을 그린다 """
        for agent in range(self.NumAgents):
            for y_offset in range(self.Size[agent]):
                for x_offset in range(self.Size[agent]):
                    index = y_offset * self.Size[agent] + x_offset
                    val = self.Data[agent][index]
                    if 0 <= y_offset + self.Ypos[agent] < HEIGHT and\
                            0 <= x_offset + self.Xpos[agent] < WIDTH and val != 0:
                        f_xpos = PIECE_GRID_SIZE + (x_offset + self.Xpos[agent]) * PIECE_GRID_SIZE
                        f_ypos = PIECE_GRID_SIZE + (y_offset + self.Ypos[agent]) * PIECE_GRID_SIZE
                        pygame.draw.rect(self.Screen, self.Color[self.MinoName[self.Current[agent]]],
                                         (f_xpos + (600 * agent),
                                          f_ypos,
                                          PIECE_SIZE,
                                          PIECE_SIZE))

    def draw_game_field(self):
        """ TODO : 필드를 그린다 """
        for agent in range(self.NumAgents):
            for y_offset in range(HEIGHT):
                for x_offset in range(WIDTH):
                    val = self.Field[agent][y_offset][x_offset]
                    color = [0, 0, 0]
                    if val != 0:
                        color = [125, 125, 125]
                    pygame.draw.rect(self.Screen,
                                     color,
                                     (PIECE_GRID_SIZE + x_offset * PIECE_GRID_SIZE + (600 * agent),
                                      PIECE_GRID_SIZE + y_offset * PIECE_GRID_SIZE,
                                      PIECE_SIZE,
                                      PIECE_SIZE))

    def draw_next_block(self):
        """ TODO : 다음 블록을 그린다 """
        ## 블록의 조각(piece)의 데이터를 구한다.
        for agent in range(self.NumAgents):
            data_type = self.MinoData[self.MinoName[self.Preview[agent][0]]]
            data = data_type[0]
            size = int_sqrt(data)
            for y_offset in range(size):
                for x_offset in range(size):
                    index = y_offset * size + x_offset
                    val = data[index]
                    # if 0 <= y_offset + self.ypos < HEIGHT and \
                    #   0 <= x_offset + self.xpos < WIDTH and
                    if val != 0:  ## 이 조건은 중요함! 0까지 그림을 그린다면, 쌓인 블록이 순간적으로 검정색이 됨.
                        ## f_xpos = filed에서의 xpos를 계산함
                        f_xpos = 460 + (x_offset) * PIECE_GRID_SIZE
                        f_ypos = 100 + (y_offset) * PIECE_GRID_SIZE
                        pygame.draw.rect(self.Screen, self.Color[self.MinoName[self.Preview[agent][0]]],
                                         (f_xpos + (600 * agent),
                                          f_ypos,
                                          PIECE_SIZE,
                                          PIECE_SIZE))

    def draw_score(self):
        """ TODO : 점수를 표시한다. """
        for agent in range(self.NumAgents):
            score_str = str(self.Score[agent]).zfill(6)
            score_image = self.SmallFont.render(score_str, True, (0, 255, 0))
            self.Screen.blit(score_image, (500 + (600 * agent), 30))

    def draw_gameover_message(self, agent_idx):
        """ TODO : 'Game Over' 문구를 표시한다 """
        score_image = self.SmallFont.render("GAME OVER", True, (255, 0, 0))
        self.Screen.blit(score_image, (200 + (600 * agent_idx), 300))

    # endregion
    # region play module
    def drop_block(self, xpos, agent_idx):
        ypos = 0
        for idx in range(0, 20):
            rtn = self.is_overlapped(xpos, 20 - idx, self.Turn[agent_idx], agent_idx)
            if not rtn:
                ypos = 20 - idx
                break
        return ypos

    def runGame(self, count):

        self.Clock.tick(10)
        self.Screen.fill((0, 0, 0))
        whole_game_over = 0
        key = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                key = event.key
            elif event.type == pygame.KEYUP:
                key = None
        for agent in range(self.NumAgents):
            game_over = self.is_game_over(agent)
            if not game_over:
                count += 5
                if count % 1000 == 0:
                    interval = max(1, 40 - 2)
                erased = self.update(count, agent)

                if erased > 0:
                    self.Score[agent] += (2 ** erased) * 100

                # 키 이벤트 처리
                next_x, next_y, next_t = \
                    self.Xpos[agent], self.Ypos[agent], self.Turn[agent]
                if key == pygame.K_UP:
                    next_t = (next_t + 1) % 4
                elif key == pygame.K_RIGHT:
                    next_x += 1
                elif key == pygame.K_LEFT:
                    next_x -= 1
                elif key == pygame.K_DOWN:
                    next_y += 1
                elif key == pygame.K_SPACE:
                    next_y = self.drop_block(next_x, agent)

                if not self.is_overlapped(next_x, next_y, next_t, agent):
                    self.Xpos[agent] = next_x
                    self.Ypos[agent] = next_y
                    self.Turn[agent] = next_t
                    self.Data[agent] = self.Type[agent][self.Turn[agent]]

            # 게임필드 그리기
            self.draw_game_field()
            # 낙하 중인 블록 그리기
            self.draw()
            # 다음 블록 그리기
            self.draw_next_block()
            # 점수 나타내기
            self.draw_score()
            # 게임 오버 메시지
            if game_over:
                whole_game_over += 1
                self.draw_gameover_message(agent)

        pygame.display.update()

        if whole_game_over >= 3:
            game_over = True
        else:
            game_over = False
        return game_over, count

    # endregion
    def step(self, actions):
        count = 0
        while True:
            terminated, count = self.runGame(count)
            if terminated:
                break
        reward = 0
        return reward, terminated, {}
