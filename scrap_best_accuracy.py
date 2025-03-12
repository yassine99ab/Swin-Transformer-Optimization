import re
import matplotlib.pyplot as plt
from collections import defaultdict

# File paths
log_file_path = "ga_validator_100.log"
output_file_path = "best_accuracy_hybrid_ga_100pop.txt"

# Initialize storage for generations
# generations = defaultdict(lambda: {'training_accuracies': [], 'validation_accuracies': [], 'weights': [], 'best_accuracy': None})


# Used to Scrap Format 1 of Logs
# current_generation = None

# # Parse the log file
# with open(log_file_path, 'r') as file:
#     for line in file:
#         # Check for generation logs
#         if "genetic_algorithm_evolution" in line:
#             match = re.search(r"Generation (\d+): ([\d.]+)%", line)
#             if match:
#                 current_generation = int(match.group(1))
#                 generations[current_generation]['best_accuracy'] = float(match.group(2))
#         elif "model_validator" in line and current_generation is not None:
#             # Extract model validator accuracies
#             match = re.search(r"Training Accuracy: ([\d.]+)%, Validation Accuracy: ([\d.]+)%", line)
#             if match:
#                 training_accuracy = float(match.group(1))
#                 validation_accuracy = float(match.group(2))
#                 generations[current_generation]['training_accuracies'].append(training_accuracy)
#                 generations[current_generation]['validation_accuracies'].append(validation_accuracy)
#                 generations[current_generation]['weights'].append(1)  # Assuming equal weights for all entries

# # Extract best accuracies by generation
# best_accuracies = []
# for gen, data in sorted(generations.items()):
#     if data['best_accuracy'] is not None:
#         best_accuracies.append(data['best_accuracy'])

# # Save best accuracies to a text file
# with open(output_file_path, 'w') as f:
#     for gen, accuracy in enumerate(best_accuracies, start=1):
#         f.write(f"Generation {gen}: {accuracy:.2f}%\n")

# print(f"Best accuracies saved to {output_file_path}")

## Used to scrap Format 2 of logs 
# best_accuracies = []
# current_generation = 0

# # Parse the log file
# with open(log_file_path, 'r') as file:
#     for line in file:
#         # Check for "Best CMA-ES accuracy" to indicate a new generation
#         if "Best CMA-ES accuracy" in line:
#             match = re.search(r"Best CMA-ES accuracy: ([\d.]+)%", line)
#             if match:
#                 current_generation += 1
#                 best_accuracy = float(match.group(1))
#                 best_accuracies.append((current_generation, best_accuracy))

# # Save best accuracies to a text file
# with open(output_file_path, 'w') as f:
#     for gen, accuracy in best_accuracies:
#         f.write(f"Generation {gen}: {accuracy:.2f}%\n")

# print(f"Best accuracies saved to {output_file_path}")

### Used to Scrap format 4 or Normal Training Logs
import re
best_accuracies = []
current_generation = 0
# File path for the log file
log_file_path = "grad_descent.log"  # Replace with your actual log file name
output_file_path = "gradient_descent.txt"

# Open the log file and the output file
with open(log_file_path, 'r') as log_file, open(output_file_path, 'w') as output_file:
    for line in log_file:
        # Match lines with training accuracy
        match = re.search(r"Train Acc: ([\d.]+)", line)
        if match:
            current_generation += 1
            train_acc = float(match.group(1))  # Extract the training accuracy
            output_file.write(f"Generation {current_generation}: {train_acc*100: .2f}%\n")

print(f"Training accuracies have been saved to {output_file_path}.")