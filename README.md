README is Work-In-Progress.

See:
https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944
On how to run

See Adx_Research.pdf for the accompanying research project. 

Derby is a simple bidding, auction, and *market* framework for creating
and running auction or market games. Environments in derby can be
interfaced in a similar fashion as environments in OpenAI’s gym:

``` python
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

A *market* can be thought of as a stateful, repeated auction:

-   A market is initialized with *m* bidders, each of which has a state.

-   A market lasts for *N* days.

-   Each day, auction items are put on sale. Each day, the bidders
    participate in an auction for the available items.

-   Each bidder’s state is updated at the end of every day. The state
    can track information such as auction items bought and amount spent.
