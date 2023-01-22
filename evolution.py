from player import Player
import numpy as np
from numpy import random
from config import CONFIG
import copy
import math

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

    def crossover(self, parent1_vector, parent2_vector, child1_vector, child2_vector):
        size = parent1_vector.shape
        crossover_point = math.floor(size[0]/2)

        # change vector in child 1 
        child1_vector[ : crossover_point] = parent1_vector[ : crossover_point]
        child1_vector[crossover_point : ] = parent2_vector[crossover_point : ]
        # change vector in child 2
        child2_vector[ : crossover_point] = parent2_vector[ : crossover_point]
        child2_vector[crossover_point : ] = parent1_vector[crossover_point : ]

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
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

            new_players = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1  = Player(self.mode)
                child2  = Player(self.mode)
                
                self.crossover(parent1.nn.W_layer1, parent2.nn.W_layer1, child1.nn.W_layer1, child2.nn.W_layer1)
                self.crossover(parent1.nn.W_layer2, parent2.nn.W_layer2, child1.nn.W_layer2, child2.nn.W_layer2)
                self.crossover(parent1.nn.b_layer1, parent2.nn.b_layer1, child1.nn.b_layer1, child2.nn.b_layer1)
                self.crossover(parent1.nn.b_layer2, parent2.nn.b_layer2, child1.nn.b_layer2, child2.nn.b_layer2)

                self.mutate(child1)
                self.mutate(child2)

                new_players.append(child1)
                new_players.append(child2)

            return new_players

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
