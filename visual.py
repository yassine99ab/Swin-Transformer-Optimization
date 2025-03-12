import matplotlib.pyplot as plt

# File paths for each algorithm's best accuracies
files = {
    "CMA-ES + DE": "best_accuracy_cmaesde_50pop.txt",
    "GA": "best_accuracy_ga_50pop.txt",
    "Hybrid GA 50 Pop": "best_accuracy_hybrid_ga_50pop.txt",
    "Hybrid GA 100 Pop": "best_accuracy_hybrid_ga_100pop.txt",
    "Gradient Descent": "gradient_descent.txt",
    "Hybrid GA With NSGA": "best_accuracies_nsga.txt"
}

# Dictionary to store data for each algorithm
accuracy_data = {}

# Read data from each file
for algorithm, file_path in files.items():
    generations = []
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            # Extract generation and accuracy from each line
            parts = line.strip().split(":")
            generation = int(parts[0].split()[1])  # Extract generation number
            accuracy = float(parts[1].strip().replace("%", ""))  # Extract accuracy
            generations.append(generation)
            accuracies.append(accuracy)
    accuracy_data[algorithm] = (generations, accuracies)

# Plot the data
plt.figure(figsize=(12, 6))
for algorithm, (generations, accuracies) in accuracy_data.items():
    plt.plot(generations, accuracies, label=algorithm)

# Graph details
plt.xlabel("Generation")
plt.ylabel("Best Training Accuracy (%)")
plt.title("Comparative Best Accuracies Across Algorithms")
plt.legend()
plt.grid(True)
plt.show()