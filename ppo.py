"""Quick demo of a two-way highway with 4 lanes (2 per direction).

Run this file to open a small viewer and watch a random agent drive on the
customised two-way environment from `highway-env`/Gymnasium. Make sure the
packages are installed first:

	pip install gymnasium highway-env
"""

import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env  # noqa: F401  # Registers the env with Gymnasium
from highway_env import utils
from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


# Lanes count is per direction in the two-way environment, so 2 = 4 total lanes.
ENV_CONFIG = {
	"lanes_count": 4,
	"vehicles_count": 25,
	"duration": 40,  # [s]
	"controlled_vehicles": 1,
	"ego_spacing": 2,
	"collision_reward": -1.0,
	"high_speed_reward": 0.4,
	"right_lane_reward": 0.1,
	"offroad_terminal": True,
	"road_length": 800,
}


class TwoWay4LaneEnv(TwoWayEnv):
	"""Two-way road with two lanes per direction and overtaking allowed."""

	@classmethod
	def default_config(cls) -> dict:
		config = super().default_config()
		config.update(ENV_CONFIG)
		return config

	def _make_road(self, length: int | None = None):  # type: ignore[override]
		length = length or self.config.get("road_length", 800)
		w = StraightLane.DEFAULT_WIDTH
		net = RoadNetwork()

		# Forward lanes (a -> b)
		# Lane 0: Right/Slow
		net.add_lane(
			"a",
			"b",
			StraightLane(
				[0, 0],
				[length, 0],
				line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
			),
		)
		# Lane 1: Left/Fast
		net.add_lane(
			"a",
			"b",
			StraightLane(
				[0, w],
				[length, w],
				line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
			),
		)
		# Lane 2: Oncoming Fast (Ghost lane for ego)
		net.add_lane(
			"a",
			"b",
			StraightLane(
				[0, 2 * w],
				[length, 2 * w],
				line_types=(LineType.NONE, LineType.STRIPED),
			),
		)
		# Lane 3: Oncoming Slow (Ghost lane for ego)
		net.add_lane(
			"a",
			"b",
			StraightLane(
				[0, 3 * w],
				[length, 3 * w],
				line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
			),
		)

		# Opposite direction (b -> a) - Occupied by oncoming traffic
		# Lane 0: Fast (Overlaps with a->b Lane 2)
		net.add_lane(
			"b",
			"a",
			StraightLane(
				[length, 2 * w],
				[0, 2 * w],
				line_types=(LineType.NONE, LineType.NONE),
			),
		)
		# Lane 1: Slow (Overlaps with a->b Lane 3)
		net.add_lane(
			"b",
			"a",
			StraightLane(
				[length, 3 * w],
				[0, 3 * w],
				line_types=(LineType.NONE, LineType.NONE),
			),
		)

		road = Road(
			network=net,
			np_random=self.np_random,
			record_history=self.config.get("show_trajectories", False),
		)
		self.road = road

	def _make_vehicles(self) -> None:  # type: ignore[override]
		road = self.road
		# Ego starts in a->b lane 1 (Fast forward)
		ego_vehicle = self.action_type.vehicle_class(
			road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
		)
		ego_vehicle.enable_lane_change = True
		road.vehicles.append(ego_vehicle)
		self.vehicle = ego_vehicle

		vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

		# Traffic in ego direction (a->b lanes 0 and 1)
		for i in range(6):
			# Randomly pick lane 0 or 1
			lane_idx = self.np_random.integers(0, 2)
			lane = ("a", "b", lane_idx)
			road.vehicles.append(
				vehicles_type(
					road,
					position=road.network.get_lane(lane).position(
						60.0 + 60.0 * float(i) + 5.0 * self.np_random.normal(), 0.0
					),
					heading=road.network.get_lane(lane).heading_at(0.0),
					speed=20.0 + 2.0 * self.np_random.normal(),
					enable_lane_change=True,
				)
			)

		# Oncoming traffic (b->a lanes 0 and 1)
		for i in range(6):
			# Randomly pick lane 0 or 1
			lane_idx = self.np_random.integers(0, 2)
			lane = ("b", "a", lane_idx)
			v = vehicles_type(
				road,
				position=road.network.get_lane(lane).position(
					200.0 + 90.0 * float(i) + 10.0 * self.np_random.normal(), 0
				),
				heading=road.network.get_lane(lane).heading_at(0.0),
				speed=22.0 + 3.0 * self.np_random.normal(),
				enable_lane_change=False,
			)
			v.target_lane_index = lane
			road.vehicles.append(v)

	def _is_truncated(self) -> bool:  # type: ignore[override]
		# Episode ends when ego reaches the end of its lane.
		lane = self.vehicle.lane
		s, _ = lane.local_coordinates(self.vehicle.position)
		return s >= lane.length - 5.0


# Register the custom env once so gym.make can build it.
register(
	id="two-way-4lane-v0",
	entry_point=TwoWay4LaneEnv,
)


def make_env(render_mode: str = "human") -> gym.Env:
	"""Create a two-way highway environment with 2 lanes in each direction."""

	# Gym wraps the env in TimeLimit; configure the underlying env instance.
	env = gym.make("two-way-4lane-v0", render_mode=render_mode, config=ENV_CONFIG)
	return env


def rollout(env: gym.Env, steps: int = 40000) -> None:
	"""Run a short random-policy rollout for visual inspection."""

	obs, info = env.reset()
	for _ in range(steps):
		print('Step:', _)
		action = env.action_space.sample()
		obs, reward, terminated, truncated, info = env.step(action)
		if terminated or truncated:
			print('Episode ended. Resetting environment.')
			obs, info = env.reset()
	env.close()


def train(total_timesteps: int = 100000, model_path: str | None = None, log_every: int = 10000):
	"""Train a PPO agent so behavior is not random, with progress prints."""

	try:
		from stable_baselines3 import PPO
		from stable_baselines3.common.vec_env import DummyVecEnv
		from stable_baselines3.common.callbacks import BaseCallback
	except ImportError as exc:  # pragma: no cover - runtime dependency
		raise SystemExit(
			"stable-baselines3 is required for training. Install with: pip install stable-baselines3"
		) from exc

	class ProgressCallback(BaseCallback):
		def __init__(self, interval: int = 10000):
			super().__init__()
			self.interval = max(1, interval)

		def _on_step(self) -> bool:
			current = min(self.num_timesteps, total_timesteps)
			if current % self.interval == 0:
				print(f"Training progress: {current}/{total_timesteps} steps")
			# Hard stop if we ever exceed the target due to rollout padding
			return current < total_timesteps

	def _builder():
		return make_env(render_mode=None)

	vec_env = DummyVecEnv([_builder])
	# Keep rollout length modest to limit overshoot beyond total_timesteps
	n_steps = max(128, min(1024, total_timesteps // 4 or 128))
	model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=n_steps)
	model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(log_every))
	if model_path:
		model.save(model_path)
	return model


def evaluate(model, episodes: int = 5):
	"""Render a trained model for a few episodes."""

	env = make_env(render_mode="human")
	for ep in range(episodes):
		obs, info = env.reset()
		terminated = truncated = False
		while not (terminated or truncated):
			action, _ = model.predict(obs, deterministic=True)
			obs, reward, terminated, truncated, info = env.step(action)
		print(f"Episode {ep + 1} finished")
	env.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Two-way 4-lane highway demo")
	parser.add_argument("--train", action="store_true", help="Train PPO instead of random rollout")
	parser.add_argument("--timesteps", type=int, default=100000, help="Training steps for PPO")
	parser.add_argument("--log-every", type=int, default=10000, help="Print progress every N steps during training")
	parser.add_argument("--model-path", type=str, default="ppo_highway.zip", help="Where to save/load model")
	parser.add_argument("--no-render", action="store_true", help="Skip rendering during eval")
	args = parser.parse_args()

	if args.train:
		model = train(total_timesteps=args.timesteps, model_path=args.model_path, log_every=args.log_every)
		if not args.no_render:
			evaluate(model, episodes=3)
	else:
		#from stable_baselines3 import PPO
		#model = PPO.load(args.model_path)
		#evaluate(model, episodes=3)
		env = make_env(render_mode="human")
		rollout(env, steps=500)
