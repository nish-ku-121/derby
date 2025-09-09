
## Project Overview

Derby is a bidding, auction, and *market* framework for creating and running auction or market games. Environments in Derby can be interfaced in a similar fashion as environments in OpenAI’s gym.

Example usage:
```python
env = ...
agents = ...
env.init(agents, num_of_days)
for i in range(num_of_trajs):
        all_agents_states = env.reset()
        for j in range(horizon_cutoff):
            actions = []
            for agent in env.agents:
                agent_states = env.get_folded_states(
                                    agent, all_agents_states
                                )
                actions.append(agent.compute_action(agent_states))
            all_agents_states, rewards, done = env.step(actions)
```

---

## Game Description

A *market* can be thought of as a stateful, repeated auction:

-   A market is initialized with *m* bidders, each of which has a state.
-   A market lasts for *N* days.
-   Each day, auction items are put on sale. Each day, the bidders participate in an auction for the available items.
-   Each bidder’s state is updated at the end of every day. The state can track information such as auction items bought and amount spent.

---

## First Time Install

This project is designed to run using Docker. You only need Docker installed—no other dependencies are required on your host machine.

1. **Install Docker**  
    Download and install Docker from:  
    https://www.docker.com/products/docker-desktop

2. **Build the Docker image**  
    In the project root directory, run:
    ```bash
    make docker-build
    ```

## Running Experiments

To run an experiment (for example, the `one_camp_n_days` experiment), use:
```bash
make docker-run ARGS="poetry run python -m derby.experiments.one_camp_n_days <experiment_name> <num_agents> <num_days> <horizon> <learning_rate>"
```
Replace the arguments as needed for your experiment.

## Rebuilding the Poetry Lock File

If you change dependencies in `pyproject.toml`, you may want to regenerate the `poetry.lock` file. Use the following make command:
```bash
make docker-lockfile
```
This will update `poetry.lock` to match the dependencies in `pyproject.toml` using Docker for a fully reproducible environment.

---

## Project Background

Derby was created by [Nishant Kumar](https://github.com/nish-ku-121) for use in his grad school research project (in collaboration with Prof. Amy Greenwald and fellow student [Enrique Areyan](https://github.com/eareyan)).

See [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf) for a brief summary.

The goal of the project was to apply (deep) reinforcement learning to the _AdX Game_. The AdX Game crudely models the digital advertising domain: advertisers buy _impression opportunities_ from websites, where the objective of each advertiser is to minimize spend and the objective of each website is to maximize revenue. This buying and selling is usually done through an _ad exchange_ (e.g. Google's AdX), which canonically holds digital auctions; the bidders are advertisers and the goods being sold are impression opportunities. In the AdX Game, each player plays the role of an _advertiser liaison_: advertisers procure _ad campaigns_ to liaisons, who are responsible for fulfilling the campaign within a certain time frame. The goal of each player is to learn what bids to place in order to maximize their profit by the end of the game.

(See pages 2 to 3 of [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf) for the game definition)


---

## RL Challenges

The AdX game is interesting to tackle from a reinforcement learning perspective because it poses several interesting properties and challenges:
- Stochasticity in the game can be a consequence of both the randomness of an impression opportunity's demographic(s) and the randomness of each player's strategy (i.e. players playing mixed strategies).
- Determining an optimal policy via _planning_ is difficult because determining a model _a priori_ is difficult or infeasible.
- The domain offers continuous control, as bids are real-valued. Furthermore, the domain can be highly dimensional, as there can be many types of demographics. Consequently exhaustive search of the space is often infeasible or intractable, thus smart exploration and/or generalization is required.
- The domain can be examined from both a single-agent perspective and a multi-agent perspective.

(See pages 8 to 11 of [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf) for an RL formulation)


---

## Algorithms

Algorithms derived, tested, and tuned include:
- Multi-Agent REINFORCE (with and without baseline)
- Multi-Agent Actor-Critic Q (with baseline)
- Multi-Agent Actor-Critic TD (has baseline)
- Multi-Agent Actor-Critic SARSA (with baseline)

(See pages 8 to 26 of [AdX RL Research Full](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Full.pdf) for algorithms and their derivations)


For all algorithms, the policy network learns a Gaussian distribution using a neural net architecture.

For algorithms that learn Q or V, the value network has a standard setup: one or more dense layers taking state (and action for Q) as input and returning a Q or V value as output. Assume ReLU activation functions.

Challenges, tips, and tricks can be found on pages 28 to 34 of [AdX RL Research Full](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Full.pdf).

---

## Results

Some results can be found:
- Pages 5 to 6 of [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf)
- Page 35 of [AdX RL Research Full](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Full.pdf)
- Pages 4 to 11 of [AdX Research Select Results](https://github.com/nish-ku-121/derby/blob/master/AdX%20RL%20Research%20Select%20Results.pdf)
