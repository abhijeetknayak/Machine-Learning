[**CS234**](http://web.stanford.edu/class/cs234/index.html) : **Reinforcement Learning** <br>

__Lecture 1 : Introduction to RL__
*Overview of RL* : <br>
  1. The agent learns and optimizes it decisions(sequences) in return for rewards. The decision making process involves **Optimization**, **Delayed consequences**, **Exploration** and **Generalization**. <br>
  2. Decisions made now might affect how we fare in the future. This is what "Delayed consequences" mean! <br>
  3. The agent has to learn by exploring its world. It gains experience, which can later be used to make good decisions. <br>
  4. **Policy** is a mapping between past experinces and decisions. The agent uses a policy to determine what decisions it has to make. Pre-programming the policy would be an option **only if** experiences that the agent needs to learn is limited. <br>
  5. Imitation Learning is almost the same as RL, where an agent learns from experiences of 'other agents'. This makes RL a supervised learning technique, and gets rid of explorations that the agent had to perform in RL, but generating training data for this is expensive. <br> 

**Sequential Decision Process** : <br>
  1. Agent performs an action first. The 'world' then provides an observation and a reward to the agent. The agent needs to keep a track of its actions, observations and rewards.<br>
  2. The agent has access to a subset of the real world. The state space is a function of the history of the agent.<br>
  3. The Markov Assumption is used here. It states that the future is independent of the past states as long as the present state is representative of the past states. <br>
  4. The agent can be modeled as a **Markov Decision Process(Full Observability(state = current observation))** or as a **Partially Observable Markov Decision Process(state = history)** <br>
  5. **Bandit** is a type of decision process where the current action does not affect the future observations. <br>

**RL Algorithm Components** : <br>
  1. *Model* - **State Model** determines the new state of the agent given a current state S<sub>t</sub> and an action A<sub>t</sub>. The resulting information is a probabilistic distribution over all states. **Reward Model** determines the immediate rewards received because of the action. The Model may be wrong though. <br>
  2. *Policy* - Helps the agent determine what action it needs to take given that it's in a particulat state. If the agent has a Deterministic policy, there's just one action the agent can take. If the policy is Stochastic, a distribution over actions is received. <br>
  3. *Value* - Value function gives the agent the expected sum of future rewards under a certain policy. The future rewards are all weighted by a certain "Discount factor". if the discount factor is zero, it means that the agent is just concerned with the immediate rewards. <br>
**Exploration and Exploitation** - When an agent performs an action trying to learn and experience new states, the agent is said to be exploring. When the agent already knows the states surrounding it and knows what best actions it needs to take, it is said to be exploiting. The agent only experiences the outcomes for the action it performs. <br>
**Evaluation and Control** - Predict the goodness of a certain policy. **Control** means optimization of the policy. Therefore, continuously evaluate and control policies! <br>
