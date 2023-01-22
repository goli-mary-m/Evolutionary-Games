from player import Player
import numpy as np
from numpy import random
from config import CONFIG
import copy


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        W_shape1 = child.nn.W_layer1.shape
        W_shape2= child.nn.W_layer2.shape

        b_shape1 = child.nn.b_layer1.shape
        b_shape2 = child.nn.b_layer2.shape

        gaussian_noise = 0.8
        child.nn.W_layer1 += random.normal(0, gaussian_noise, W_shape1)
        child.nn.W_layer2 += random.normal(0, gaussian_noise, W_shape2)

        child.nn.b_layer1 += random.normal(0, gaussian_noise, b_shape1)
        child.nn.b_layer2 += random.normal(0, gaussian_noise, b_shape2)        
        return child

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            new_child = []
            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

            'method: Top K Selection'
            parents = self.top_k_algorithm(prev_players, num_players)

            'method: Rolette Wheel: select parents based on fitness proportionate'
            # parents = self.rolette_wheel_algorithm(prev_players, num_players)
           
            'method: SUS'
            # parents = self.stochastic_universal_sampling_algorithm(prev_players, num_players)

            pairs = list(zip(parents[::2], parents[1::2]))
            for pair in pairs:
                crossover_children = self.crossover(*pair)
                mutated_cross_children = self.mutate(crossover_children)
                new_child.append(mutated_cross_children)
            # generate children
            # new_players = (parents).copy()
            # for player in new_players:
            #     new_child.append(self.mutate(player))

            return new_child

    def crossover(self, player1, player2):
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player1.nn)
        new_player.fitness = player1.fitness

        W_shape1 = new_player.nn.W_layer1.shape
        b_shape1 = player2.nn.b_layer1.shape

        W_shape2 = new_player.nn.W_layer2.shape
        b_shape2 = player2.nn.b_layer2.shape

        player1_W1 = player1.nn.W_layer1.shape
        player1_b1 = player2.nn.b_layer1.shape

        player1_W2 = player1.nn.W_layer2.shape
        player1_b2 = player2.nn.b_layer2.shape

        player2_W1 = player1.nn.W_layer1.shape
        player2_b1 = player2.nn.b_layer1.shape

        player2_W2 = player1.nn.W_layer2.shape
        player2_b2 = player2.nn.b_layer2.shape

        # for i in range(W_shape1[1]):
            # if random.uniform(0, 1) > 0:
                # if i % 4 < 2:
                    # new_player.nn.W_layer1[:, i] = player1_W[:, i]
                # else:
                    # new_player.nn.W_layer1[:, i] = player1_W[:, i]

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        
        'Implementing top-k algorithm'
        # selected_next_population = self.top_k_algorithm(players, num_players)
        
        'Implement roulette wheel'
        selected_next_population = self.rolette_wheel_algorithm(players, num_players)

        'Implement SUS'
        # selected_next_population = self.stochastic_universal_sampling_algorithm(players, num_players)

        # Implement Q-tournament
        return selected_next_population
    
    def top_k_algorithm(self, players, num_players):
        sorted_players = sorted(players, key=lambda x: x.fitness, reverse=True)
        top_k_players = sorted_players[:num_players]
        return top_k_players
    
    def rolette_wheel_algorithm(self, players, num_players):
        max_fitness = sum([p.fitness for p in players])         # Computes the totallity of the population fitness
        players_probs = [p.fitness/max_fitness for p in players]           # Computes for each chromosome the probability 
        selected_indices = np.random.choice(len(players), num_players, p=players_probs)
        selected_players = [players[i] for i in selected_indices]
        return selected_players

    def stochastic_universal_sampling_algorithm(self, players, num_players):
        selected_players = []
        max_fitness = sum([p.fitness for p in players])
        players_probs = [p.fitness/max_fitness for p in players]

        step_size = 1.0/num_players
        r = random.uniform(0, step_size)

        cumulative_prob = 0
        for i in range(len(players)):
            players_prob = players_probs[i]
            cumulative_prob += players_prob
            if r <= cumulative_prob:
                selected_players.append(players[i])
                r += step_size
        
        if len(selected_players) < num_players:
            selected_players.append(players[-1])

        return selected_players
