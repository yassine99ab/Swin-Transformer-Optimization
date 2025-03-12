import numpy as np
import torch
from deap import base, creator, tools
import os
import logging
class GeneticAlgorithmNSGA:
    def __init__(self, min_weight, max_weight, last_layer_shape, validation_class):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.last_layer_shape = last_layer_shape
        self.validation_class = validation_class
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='nsga_run.log', level=logging.INFO)
        # DEAP Setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
    "attr_float",
    lambda: np.clip(np.random.normal(loc=0.0, scale=0.05), self.min_weight, self.max_weight)
)
        self.toolbox.register(
            "individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=np.prod(last_layer_shape)
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register Genetic Algorithm operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.5)
        self.toolbox.register("select", tools.selRoulette)

        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize accuracy, minimize regularization
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox.register("attr_float", np.random.uniform, self.min_weight, self.max_weight)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=np.prod(last_layer_shape))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selNSGA2)

    def levy_flight(self, step_size=0.01, beta=1.5):
        """Generate a Lévy flight step."""
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        return step_size * u / abs(v)**(1 / beta)

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

    def cuckoo_search_with_sa(self, population, step_size=0.04, fraction=0.25, temperature=1.0):
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

    def fitness_function(self, individual):
        """Evaluate fitness by setting weights and computing accuracy."""
        individual_tensor = torch.FloatTensor(individual).reshape(self.last_layer_shape)
        accuracy = self.validation_class.validate_model_with_weights(individual_tensor)  # Objective 1
        regularization = torch.sum(individual_tensor**2).item() / 1e6 # Gaussian regularizer (Objective 2)
        return accuracy, regularization


    def _evaluate_population_with_cuda_streams(self, population, generation , total_generations):
        """Evaluate fitness of the population using CUDA streams."""
        streams = [torch.cuda.Stream(device=self.validation_class.device) for _ in range(len(population))]
        fitness_results = [None] * len(population)

        # Start evaluation in parallel streams
        for i, (ind, stream) in enumerate(zip(population, streams)):
            with torch.cuda.stream(stream):
                weights = torch.FloatTensor(ind).reshape(self.last_layer_shape).to(self.validation_class.device)
                fitness_results[i] = self.validation_class.validate_model_with_weights(weights, generation, total_generations)

        # Synchronize all streams
        torch.cuda.synchronize(self.validation_class.device)
        return fitness_results

    def run(self, population_size=100, num_generations=50, stop_threshold=99, checkpoint_file="nsga_checkpoint.pt", save_interval=4):
        """Run NSGA-II for the bi-objective optimization with Pareto front-based stopping criterion."""
        self.toolbox.register("evaluate", self.fitness_function)

        # Initialize population
        population = self.toolbox.population(n=population_size)

        # Resume from checkpoint if it exists
        start_generation = 1
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            population = checkpoint["population"]
            start_generation = checkpoint["generation"] + 1
            print(f"Resuming from generation {start_generation}.")

        pareto_front = []

        for generation in range(start_generation, num_generations + 1):
            print(f"Generation {generation}")

            # Evaluate accuracy using the helper method
            fitnesses = self._evaluate_population_with_cuda_streams(population, generation, num_generations)

            # Assign the bi-objective fitness (accuracy and regularization) to each individual
            for ind, accuracy in zip(population, fitnesses):
                # Calculate Gaussian regularization (sum of squared weights)
                weights = torch.FloatTensor(ind).reshape(self.last_layer_shape)
                regularization = torch.sum(weights**2).item() / 1e6

                # Assign bi-objective fitness
                ind.fitness.values = (accuracy, regularization)

            # Extract Pareto front
            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

            # Check stopping criterion on Pareto front
            for ind in pareto_front:
                if ind.fitness.values[0] >= stop_threshold:  # Check if accuracy meets threshold
                    print(f"Stopping criterion reached at generation {generation}.")
                    self.logger.info(f"Stopping criterion reached with best individual: {ind.fitness.values}")
                    return ind  # Return the best individual meeting the criterion

             # Check for stopping criterion
            best_ind = tools.selBest(population, k=1)[0]
            best_fitness = best_ind.fitness.values[0]
            self.logger.info(f"Best accuracy in Generation {generation}: {best_fitness:.2f}%")

            
            # Select the next generation using NSGA-II
            offspring = self.toolbox.select(population, len(population) - 5)  # Reserve space for elites
            offspring = list(map(self.toolbox.clone, offspring))  # Clone selected individuals

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values  # Invalidate fitness after crossover
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if np.random.rand() < 0.4:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values  # Invalidate fitness after mutation

            # Retain elites
            elites = tools.selBest(population, k=5)  # Select top 5 elite individuals

            # Update population
            population[:] = offspring + elites
            # Save checkpoint
            if generation % save_interval == 0:
                torch.save({
                    "population": population,
                    "generation": generation,
                    "pareto_front": pareto_front,
                }, checkpoint_file)
                print(f"Checkpoint saved at generation {generation}.")

        # Return the final Pareto front if no individual meets the stopping criterion
        return tools.selBest(population, k=1)[0]

