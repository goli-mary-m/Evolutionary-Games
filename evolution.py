from player import Player
import numpy as np
from numpy import random
from config import CONFIG


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
        pass


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

            new_players = prev_players
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
        # selected_next_population = self.rolette_wheel_algorithm(players)

        'Implement SUS'
        selected_next_population = self.stochastic_universal_sampling_algorithm(players, num_players)

        # Implement Q-tournament
        return selected_next_population
    
    def top_k_algorithm(self, players, num_players):
        sorted_players = sorted(players, key=lambda x: x.fitness, reverse=True)
        top_k_players = sorted_players[:num_players]
        return top_k_players
    
    def rolette_wheel_algorithm(self, players):
        # Computes the totallity of the population fitness
        players_fitness = sum([p.fitness for p in players])

        # Computes for each chromosome the probability 
        players_probs = [p.fitness/players_fitness for p in players]

        return players[random.choice(len(players), p=players_probs)]

    def stochastic_universal_sampling_algorithm(self, players, num_players):
        selected_players = []
        max_fitness = sum([p.fitness for p in players])
        players_probs = [p.fitness/max_fitness for p in players]

        step_size = 1.0/num_players
        r = random.unifrom(0, step_size)

        cumulative_prob = 0
        for i in range(players):
            players_prob = players_probs[i]
            cumulative_prob += players_probs
            if r <= cumulative_prob:
                selected_players.append(players[i])
                r += step_size
        
        if len(selected_players) < num_players:
            selected_players.append(players[-1])

        return selected_players
