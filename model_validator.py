import torch
import logging

class ModelValidator:
    def __init__(self, model, train_loader, val_loader, device, w_train=0.5, w_val=0.5, training_emphasis=0.6, validation_emphasis=0.3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.w_train = w_train
        self.w_val = w_val
        self.training_emphasis = training_emphasis
        self.validation_emphasis = validation_emphasis
        self.logger = logging.getLogger(__name__)


    def _compute_accuracy(self, weights, data_loader):
        self.model.head.weight.data = weights.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def _normalize_weights(self, weights):
        total = sum(weights)
        return [w / total for w in weights]

    # Lots of Parameters were tried and tested but they later was scraped out in favour or simple calculation
    def validate_model_with_weights(self, weights, generation=0, total_generations=0,
                                     alpha_initial=0.9, alpha_final=0.7,
                                     beta_initisl=0.1, beta_final=0.5,
                                     gamma_initial=0.01, gamma_final=0.8):
        train_accuracy = self._compute_accuracy(weights, self.train_loader)
        val_accuracy = self._compute_accuracy(weights, self.val_loader)


        fitness_score = train_accuracy

        self.logger.info(
            f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, " )
        print(f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, ")

        return fitness_score
