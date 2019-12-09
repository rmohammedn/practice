import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld:
	""" its just model of grid world """

	def __init__(self, shape=(4,4)):
		""" initialize all gridworld parameter """
		self.shape = shape
		self.ns = np.prod(shape)
		self.na = 4
		max_x = shape[0]
		max_y = shape[1]
		grid = np.arange(self.ns).reshape(shape)
		it = np.nditer(grid, flags=["multi_index"])
		p = {}

		while not it.finished:
			s = it.iterindex
			y, x = it.multi_index

			p[s] = {a : [] for a in range(self.na)}
			is_done = lambda s : s == 0 or s == (self.ns - 1)
			reward = 0.0 if is_done(s) else -1.0

			if is_done(s):
				p[s][UP] = [(1.0, s, reward, True)]
				p[s][RIGHT] = [(1.0, s, reward, True)]
				p[s][DOWN] = [(1.0, s, reward, True)]
				p[s][LEFT] = [(1.0, s, reward, True)]
			else:
				ns_up = s if y == 0 else s - max_x
				ns_right = s if x == max_x - 1 else s + 1
				ns_down = s if y == max_y - 1 else s + max_x
				ns_left = s if x == 0 else s - 1

				p[s][UP] = [(1.0, ns_up, reward, is_done(s))]
				p[s][RIGHT] = [(1.0, ns_right, reward, is_done(s))]
				p[s][DOWN] = [(1.0, ns_down, reward, is_done(s))]
				p[s][LEFT] = [(1.0, ns_left, reward, is_done(s))]

			it.iternext()
		self.p = p

class PolicyIteration:
	""" policy iteration method of reinforcement learning """

	def __init__(self, env, epoch=1000, gamma=1, delta=0.005):
		""" initialize Policy iteration using gridworld """

		self.env = GridWorld()
		self.policy = None
		self.policy = None
		self.v = None
		self.delta = delta
		self.gamma = gamma
		self.epoch = epoch

	def policyEvaluation(self, policy):
		""" Using given policy update state value function """
		v = np.zeros(self.env.ns)
		while True:
			delta = 0.0

			for s in range(self.env.ns):
				state_value = 0

				for a, prob_a in enumerate(policy[s]):
					for prob_s, next_state, reward, _ in self.env.p[s][a]:
						state_value += prob_a * prob_s * (reward + self.gamma * v[next_state])

				delta = abs(state_value - v[s])
				v[s] = state_value

			if delta < self.delta:
				break
		self.v = v

	def policyImprovment(self, policy):
		""" improve the policy for given state value """

		for s in range(self.env.ns):
			qa = np.zeros(self.env.na)

			for a in range(self.env.na):
				for prob_s, next_state, reward, _ in self.env.p[s][a]:
					qa[a] += prob_s * (reward + self.gamma * self.v[next_state])

			best_action = np.argmax(qa)
			policy[s] = np.eye(self.env.na)[best_action]

		return policy

	def policyIteration(self):
		""" combine policy iteration """
		env = self.env
		policy = np.ones([env.ns, env.na]) / env.na

		for i in range(self.epoch):
			self.policyEvaluation(policy)
			old_policy = np.copy(policy)
			new_policy = self.policyImprovment(policy)

			if np.all(new_policy == old_policy):
				print("policy converged at %d iteration" %(i+1))
				break
			policy = new_policy

		self.policy = policy

class ValueIteration:
	""" value iteration method of reinforcement learning """
	def __init__(self, env, epoch=1000, gamma=1, delta=0.05):
		""" initializing all the parameters """

		self.env = env
		self.epoch = epoch
		self.gamma = gamma
		self.delta = delta
		self.v = np.zeros(env.ns)
		self.policy = None

	def optimalValueFunction(self):
		""" using Bellmen optimality equation """
		while True:
			for s in range(self.env.ns):
				qa = np.zeros(self.env.na)

				for a in range(self.env.na):
					for prob_s, next_state, reward, _ in self.env.p[s][a]:
						qa[a] += prob_s * (reward + self.gamma * self.v[next_state])

				max_q_value = np.max(qa)
				delta = abs(max_q_value - self.v[s])
				self.v[s] = max_q_value

			if delta < self.delta:
				break

	def optimalPolicyExtraction(self):
		""" extracting optimal policy from Bellmann optimal state equation"""

		policy = np.zeros([self.env.ns, self.env.na])
		for s in range(self.env.ns):
			qa = np.zeros(self.env.na)

			for a in range(self.env.na):
				for prob_s, next_state, reward, _ in self.env.p[s][a]:
					qa[a] += prob_s * (reward + self.gamma * self.v[next_state])

			best_action = np.argmax(qa)
			policy[s] = np.eye(self.env.na)[best_action]

		self.policy = policy

	def valueIteration(self):
		""" building value iteration """

		self.optimalValueFunction()
		self.optimalPolicyExtraction()
		#print(self.policy)


def main():
	env = GridWorld()
	#policy_method = PolicyIteration(env)
	#policy_method.policyIteration()
	#print(policy_method.policy)
	#print(policy_method.v)
	#p = policy_method.policy
	value_method = ValueIteration(env)
	value_method.valueIteration()
	print(value_method.policy)
	print(value_method.v)
	p = value_method.policy
	grid = np.zeros(env.shape, dtype='string')
	action = ['up', 'right', 'down', 'left']
	i = 0
	for i in range(4):
		for j in range(4):
			grid[i,j] = action[np.argmax(p[4*i + j])]
	print(grid)


if __name__ == "__main__":
	main()