class Measure:
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """
        compute utilities of each policy

        Args:
            states: (n_actors, d_state)
            actions: (n_actors, d_action)
            next_state_means: (n_actors, ensemble_size, d_state)
            next_state_vars: (n_actors, ensemble_size, d_state)

        Returns:
            utility: (n_actors)
        """

        raise NotImplementedError
