import numpy as np
import torch
from deap import base, creator, tools
import os
import logging
import random

class GeneticAlgorithmEvolution:
    def __init__(self, min_weight, max_weight, last_layer_shape, validation_class):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.last_layer_shape = last_layer_shape
        self.validation_class = validation_class
        self.logger = logging.getLogger(__name__)
        # DEAP Setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.combined_initialization)
        self.toolbox.register(
            "individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=np.prod(last_layer_shape)
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register Genetic Algorithm operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.6, indpb=0.5)
        # self.toolbox.register("select", tools.selRoulette)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def combined_initialization(self):
        strategy = np.random.choice(["uniform", "gaussian", "edge"], p=[0.5, 0.4, 0.1])
        if strategy == "uniform":
            return np.random.uniform(self.min_weight, self.max_weight)
        elif strategy == "gaussian":
            return np.clip(np.random.normal((self.min_weight + self.max_weight) / 2,
                                            (self.max_weight - self.min_weight) / 4),
                        self.min_weight, self.max_weight)
        elif strategy == "edge":
            return self.min_weight if np.random.rand() < 0.5 else self.max_weight
    
    def levy_flight(self, step_size=0.1, beta=1.5):
        """Generate a Lévy flight step."""
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        return step_size * u / abs(v)**(1 / beta)

    def fitness_function(self, individual):
        individual_tensor = torch.FloatTensor(individual).reshape(self.last_layer_shape).to(self.validation_class.device)
        accuracy = self.validation_class.validate_model_with_weights(individual_tensor)
        return accuracy,
    def _evaluate_population_with_cuda_streams(self, population, generation, total_generations):
        """
        Evaluate fitness of the population using CUDA streams, avoiding redundant computations.

        Args:
            population (list): The list of individuals to evaluate.
            generation (int): Current generation number.
            total_generations (int): Total number of generations.

        Returns:
            List of fitness values for the population.
        """
        streams = [torch.cuda.Stream(device=self.validation_class.device) for _ in range(len(population))]
        fitness_results = [None] * len(population)

        # Start evaluation in parallel streams
        for i, (ind, stream) in enumerate(zip(population, streams)):
            # Skip evaluation if fitness is valid
            if ind.fitness.valid:
                fitness_results[i] = ind.fitness.values[0]
            else:
                with torch.cuda.stream(stream):
                    weights = torch.FloatTensor(ind).reshape(self.last_layer_shape).to(self.validation_class.device)
                    fitness_results[i] = self.validation_class.validate_model_with_weights(weights, generation, total_generations)

        # Synchronize all streams
        torch.cuda.synchronize(self.validation_class.device)

        return fitness_results

    def cuckoo_search_with_sa(self, population, step_size=0.09, fraction=0.20, temperature=1.0):
        """Introduce cuckoo solutions using Lévy flights and apply Simulated Annealing."""
        num_cuckoos = int(len(population) * fraction)
        new_cuckoos = []

        for _ in range(num_cuckoos):
            # Pick a random solution
            random_ind = np.random.randint(len(population))
            cuckoo = population[random_ind][:]

            # Apply Lévy flight
            for i in range(len(cuckoo)):
                cuckoo[i] += self.levy_flight(step_size=step_size)
                cuckoo[i] = np.clip(cuckoo[i], self.min_weight, self.max_weight)

            # Create a candidate individual and evaluate its fitness
            candidate_ind = creator.Individual(cuckoo)
            candidate_ind.fitness.values = self.toolbox.evaluate(candidate_ind)

            # Apply Simulated Annealing to decide acceptance
            new_cuckoos.append(self.simulated_annealing(population[random_ind], candidate_ind, temperature))

        # Replace the worst-performing individuals with the accepted cuckoos
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0])
        for i in range(num_cuckoos):
            sorted_population[i][:] = new_cuckoos[i][:]

        return sorted_population

    def population_diversity(self, population):
        """
        Calculate population diversity as the mean standard deviation of the genes across all individuals.
        
        Args:
            population (list): List of individuals in the population.

        Returns:
            float: Diversity value (mean standard deviation across genes).
        """
        # Convert population to a NumPy array
        population_array = np.array([ind for ind in population])

        # Compute the standard deviation across genes
        diversity = np.mean(np.std(population_array, axis=0))
        return diversity

    def simulated_annealing(self, current_ind, candidate_ind, temperature):
        """Apply Simulated Annealing to accept or reject a candidate solution."""
        current_fitness = current_ind.fitness.values[0]
        candidate_fitness = candidate_ind.fitness.values[0]

        if candidate_fitness > current_fitness:
            return candidate_ind  # Accept better solution
        else:
            # Accept worse solution with a probability based on temperature
            prob_accept = np.exp((candidate_fitness - current_fitness) / temperature)
            if np.random.rand() < prob_accept:
                return candidate_ind  # Accept worse solution
            else:
                return current_ind  # Keep current solution
    
    def generate_individual_near_population(self, population):
        """Generate a new individual near the population mean."""
        population_array = np.array([ind for ind in population])
        mean = np.mean(population_array, axis=0)
        std = np.std(population_array, axis=0)

        # Sample from a Gaussian distribution around the mean
        new_individual = np.random.normal(mean, std).tolist()
        new_individual = np.clip(new_individual, self.min_weight, self.max_weight)
        return creator.Individual(new_individual)

    def reintroduce_diversity_with_strategies(
    self, 
    population, 
    diversity_threshold=0.01, 
    mutate_fraction=0.3, 
    crossover_fraction=0.4, 
    sampling_fraction=0.3, 
    elitism_fraction=0.2, 
    tournsize=3
):
        """
        Reintroduce diversity using a mix of aggressive mutation, sampling, and crossover strategies.

        Args:
            population (list): Current population.
            diversity_threshold (float): Threshold for triggering diversity reintroduction.
            mutate_fraction (float): Fraction of elites to mutate aggressively.
            crossover_fraction (float): Fraction of individuals generated by crossing top selections with elites.
            sampling_fraction (float): Fraction of individuals generated by sampling around population mean.
            elitism_fraction (float): Fraction of population preserved as elites.
            tournsize (int): Tournament size for selection.
        """
        original_mutate = self.toolbox.mutate

        # Temporarily register the custom mutation
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.7, indpb=0.8)
        # Calculate the number of individuals for each strategy
        population_size = len(population)
        num_elites = int(population_size * elitism_fraction)
        num_mutate = int(population_size * mutate_fraction)
        num_crossover = int(population_size * crossover_fraction)
        num_sampling = int(population_size * sampling_fraction)

        # Sort population by fitness
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        elites = sorted_population[:num_elites]

        # **1. Aggressive Mutation of Top 30% Elites**
        mutated_population = []
        for elite in elites[:num_mutate]:
            mutated_individual = self.toolbox.clone(elite)
            self.toolbox.mutate(mutated_individual)  # Apply aggressive mutation
            mutated_population.append(mutated_individual)

        # **2. Tournament Selection for Sampling Around Population Mean**
        sampled_population = []
        for _ in range(num_sampling):
            parent = tools.selTournament(population, k=1, tournsize=tournsize)[0]
            sampled_individual = self.generate_individual_near_population(population)
            sampled_population.append(sampled_individual)

        # **3. Crossover Between Top 10% and Elites**
        crossover_population = []
        for _ in range(num_crossover):
            elite_parent = random.choice(elites)
            top_parent = random.choice(sorted_population[num_elites:num_elites + num_crossover])
            child = self.toolbox.clone(elite_parent)
            self.toolbox.mate(child, top_parent)  # Perform crossover
            crossover_population.append(child)
        
        # Restore the original mutation operator
        self.toolbox.unregister("mutate")  # Unregister the temporary mutation operator
        self.toolbox.register("mutate", original_mutate)  # Re-register the original mutation operator

        print(f"Diversity reintroduced: {num_mutate} mutated, {num_sampling} sampled, {num_crossover} crossover individuals.")
        # Combine all strategies into the population
        # Calculate how many individuals to replace
        replace_count = int(len(population) * 0.8)  # Replace 50%
        retain_count = len(population) - replace_count

        # Sort by fitness and retain the top individuals
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        retained_individuals = sorted_population[:retain_count]

        # Combine retained individuals with new ones
        all_candidates =  mutated_population + sampled_population + crossover_population

        # Shuffle to randomize the order of candidates
        random.shuffle(all_candidates)

        # Ensure the population size remains constant
        population[:] = retained_individuals + all_candidates[:len(population)-len(retained_individuals)]

    def run(self, population_size=50, num_generations=50, stop_threshold=99, checkpoint_file="pure_ga_run1.pt", save_interval=4):
        """Run the GA algorithm"""
        self.toolbox.register("evaluate", self.fitness_function)
        

        # Initialize population
        population = self.toolbox.population(n=population_size)

        # Simulated Annealing parameters
        initial_temperature = 1.0
        cooling_rate = 0.95

        # Resume from checkpoint if it exists
        start_generation = 1
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            population = checkpoint["population"]
            start_generation = checkpoint["generation"] + 1
            initial_temperature = checkpoint.get("temperature", 1.0)  # Restore temperature
            print(f"Resuming from generation {start_generation}.")

        prev_best_fitness = -float("inf")
        for generation in range(start_generation, num_generations + 1):
            print(f"Generation {generation}")

            # Evaluate fitness
            fitnesses = self._evaluate_population_with_cuda_streams(population, generation, num_generations)
            for ind, fitness in zip(population, fitnesses):
                ind.fitness.values = (fitness,)

            if generation < num_generations // 1.5:
                eta = 4  # Exploration
            else:
                eta = 20  # Exploitation
            self.toolbox.unregister("mate")
            self.toolbox.register("mate", tools.cxTwoPoint)
            

            # Stopping criterion
            best_ind = tools.selBest(population, k=1)[0]
            best_fitness = best_ind.fitness.values[0]
            self.logger.info(f"Best accuracy in Generation {generation}: {best_fitness:.2f}%")

            # Check for stagnation or progress
            if abs(best_fitness - prev_best_fitness) < 0.5:
                print("No significant improvement in fitness. Adjusting mutation rate.")
                # Increase mutation to escape stagnation
                self.toolbox.unregister("mutate")
                self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.8, indpb=0.7)
            else:
                # Reset mutation to default
                self.toolbox.unregister("mutate")
                self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.5)

            prev_best_fitness = best_fitness  # Update previous best fitness
            if best_fitness >= stop_threshold:
                print("Stopping criterion reached.")
                return best_ind

            # Apply Cuckoo Search (Commented For Traditional GA, Uncomment to flip back to Hybrid)
            # population = self.cuckoo_search_with_sa(population, step_size=0.05, fraction=0.20, temperature=initial_temperature)
            initial_temperature *= cooling_rate
            # Apply elitism
            elite_count = 5 # Configurable
            elites = tools.selBest(population, k=elite_count)

            # Apply GA operations
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if np.random.rand() < 0.4:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Uncomment to flip back to Hybrid 
            # diversity = self.population_diversity(population)
            # Log diversity
            # self.logger.info(f"Generation {generation}: Diversity={diversity:.4f}") 
            # # Reintroduce diversity if needed
            # if diversity < 0.125:
            #    self.reintroduce_diversity_with_strategies(population)

            # Replace worst individuals with elites
            offspring[-elite_count:] = elites
            population[:] = offspring

            # Update temperature for Simulated Annealing
            initial_temperature *= cooling_rate

            # Save checkpoint
            if generation % save_interval == 0:
                torch.save({
                    "population": population,
                    "generation": generation,
                    "temperature": initial_temperature,
                }, checkpoint_file)
                print(f"Checkpoint saved at generation {generation}.")

        return tools.selBest(population, k=1)[0]