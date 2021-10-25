from gym import spaces
from mapgen import Dungeon


class Wrapper(Dungeon):
    def __init__(self,
                 width=64,
                 height=64,
                 max_rooms=25,
                 min_room_xy=10,
                 max_room_xy=25,
                 observation_size: int = 11,
                 vision_radius: int = 5,
                 max_steps: int = 2000
                 ):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            vision_radius=vision_radius,
            max_steps=max_steps
        )

        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3])
        # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.action_space = spaces.Discrete(3)
        self.respawn_explored = 0

    def reset(self):
        observation = super().reset()
        return observation[:, :, :-1] # remove trajectory

    def step(self, action: int):
        observation, reward, done, info = super().step(action)
        observation = observation[:, :, :-1]
        # set reward as a fraction of new explored cells (so total reward is 1.0)
        # reward = explored / self._map._visible_cells
        if info['step'] == 1:
            self.respawn_explored = info["total_explored"] - info['new_explored']

        if info['moved']:
            # не врезались
            if info['new_explored'] > 0:
                # хотим в конце быстро находить пропущенные клетки,
                # но не хотим награждать себя за открытые клетки, если ничего не открыли
                # также не будем себя хвалить за первый степ
                # если не открыли новых
                reward = 1 * reward + 5 * (info["total_explored"] - self.respawn_explored) / (info["total_cells"] - self.respawn_explored)
            if info['is_new']:
                # предпочитаем переходить к новым клеткам
                reward += 0.1
            # хотим быстрее пройти, поэтому штрафуем за действия
            if action == 1:
                # предпочитаем идти вперед, а не повороты
                reward -= 0.1
            else:
                reward -= 0.5
        else:
            # врезались
            reward = -1
        return observation, reward, done, info
