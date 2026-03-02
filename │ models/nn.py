
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
from datetime import datetime

N_SAMPLES      = 1000   # MNIST subset size
BITS_PER_WEIGHT = 8     # quantize each weight to 8-bit integer
MAX_ITERATIONS  = 5000
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#simple nn
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
#only consider subset of the data
def load_mnist_subset(n=N_SAMPLES):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    indices = list(range(n))
    subset  = torch.utils.data.Subset(dataset, indices)
    loader  = torch.utils.data.DataLoader(subset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    return images, labels   # shape looks like [n,1,28,28], [n]


def get_flattened_weights(model, bits =BITS_PER_WEIGHT):
    """
    Flatten all model weights into a 1D numpy array
    """
    flat = np.concatenate([p.data.cpu().numpy().flatten()
                           for p in model.parameters()])
    w_min, w_max = flat.min(), flat.max()
    scale = (w_max - w_min) / (2**bits - 1) if w_max != w_min else 1.0
    zero  = w_min
    flat_int = np.round((flat - zero) / scale).astype(np.int32)
    flat_int = np.clip(flat_int, 0, 2**bits - 1)
    return flat_int, scale, zero

def flat_int_to_weights(flat_int, model, scale, zero):
    """Reconstruct float weights from quantized integers and load into model."""
    flat_float = flat_int.astype(np.float32) * scale + zero
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(torch.tensor(flat_float[idx:idx+numel].reshape(p.shape)))
            idx += numel

#loss function
def compute_loss(model, images, labels):
    """Cross-entropy loss on the fixed subset."""
    model.eval()
    with torch.no_grad():
        logits = model(images)
        loss   = F.cross_entropy(logits, labels).item()
    return loss

def compute_accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
        return (preds == labels).float().mean().item()

# flip one bit of one quantized weight
def flip_bit(flat_int, weight_idx, bit_idx, bits=BITS_PER_WEIGHT):
    """Return a NEW array with bit `bit_idx` of weight `weight_idx` flipped."""
    new_arr = flat_int.copy()
    new_arr[weight_idx] = new_arr[weight_idx] ^ (1 << bit_idx)
    new_arr[weight_idx] = np.clip(new_arr[weight_idx], 0, 2**bits - 1)
    return new_arr


class RTS_NN:
    def __init__(self, model, images, labels, max_iterations=MAX_ITERATIONS):
        self.model          = model
        self.images         = images
        self.labels         = labels
        self.max_iterations = max_iterations
        self.bits           = BITS_PER_WEIGHT

        # Encode initial weights
        self.flat_int, self.scale, self.zero = get_flattened_weights(model,self.bits)
        self.n_weights  = len(self.flat_int)
        self.total_moves = self.n_weights * self.bits  # total possible (weight, bit) pairs

        # Initial cost
        self.current_f   = compute_loss(model, images, labels)
        self.best_f      = self.current_f
        self.best_int    = self.flat_int.copy()

        # Tabu memory
        self.occupation  = np.full((self.n_weights, self.bits), -np.inf)

        # RTS parameters
        self.list_size   = 1.0
        self.moving_avg  = self.n_weights / 2
        self.chaotic     = 0
        self.steps_since_change = 0
        self.current_time = 0
        self.pointer     = {}   # config hash has {last_time, repetitions}

        self.stats = {
            'escape_count': 0,
            'cycle_detections': 0,
            'aspiration_count': 0,
            'loss_history': [self.current_f],
        }

        print(f"Network weights (flattened): {self.n_weights}")
        print(f"Bits per weight            : {self.bits}")
        print(f"Total possible moves       : {self.total_moves:,}")
        print(f"Initial loss               : {self.current_f:.4f}")
        print(f"Initial accuracy           : {compute_accuracy(model, images, labels):.3f}")

    
    # mapping

    def decode_move(self, move_id):
        """move_id -> weight_idx, bit_idx by div/mod"""
        weight_idx = move_id // self.bits   # quotient
        bit_idx    = move_id  % self.bits   # remainder
        return weight_idx, bit_idx

    def encode_move(self, weight_idx, bit_idx):
        """weight_idx, bit_idx -> move_id."""
        return weight_idx * self.bits + bit_idx

   
    # Tabu check
  
    def is_tabu(self, weight_idx, bit_idx):
        return self.occupation[weight_idx][bit_idx] >= self.current_time - int(self.list_size)

    def make_tabu(self, weight_idx, bit_idx):
        self.occupation[weight_idx][bit_idx] = self.current_time

    
    # Cycle detection 
    def check_for_repetition(self, cycle_max=50, Rep=3, Chaos=3,
                              Increase=1.1, Decrease=0.9):
        self.steps_since_change += 1
        # Hash current config as a tuple of the quantized weights
        config = tuple(self.flat_int.tolist())
        max_size = self.n_weights * self.bits // 2  

        if config in self.pointer:
            length = self.current_time - self.pointer[config]['last_time']
            self.pointer[config]['last_time']     = self.current_time
            self.pointer[config]['repetitions']  += 1

            if self.pointer[config]['repetitions'] > Rep:
                self.chaotic += 1
                self.stats['cycle_detections'] += 1
                if self.chaotic > Chaos:
                    self.chaotic = 0
                    return True  # trigger escape

            if length < cycle_max:
                self.moving_avg = 0.1 * length + 0.9 * self.moving_avg
                self.list_size  = min(self.list_size * Increase, max_size)
                self.steps_since_change = 0
        else:
            self.pointer[config] = {'last_time': self.current_time, 'repetitions': 0}

        if self.steps_since_change > self.moving_avg:
            self.list_size = max(self.list_size * Decrease, 1.0)
            self.steps_since_change = 0

        return False

   
    # Choose best move 

    def choose_best_move(self, n_candidates=50):
        """
        Sample `n_candidates` random (weight_idx, bit_idx) pairs,
        evaluate each, return the best non-tabu one (aspiration can override).
        """
        best_move  = None
        best_delta = float('inf')

        # Sample random move ids
        move_ids = random.sample(range(self.total_moves), min(n_candidates, self.total_moves))

        for move_id in move_ids:
            weight_idx, bit_idx = self.decode_move(move_id)

            # Try flipping this bit
            new_int  = flip_bit(self.flat_int, weight_idx, bit_idx, self.bits)
            flat_int_to_weights(new_int, self.model, self.scale, self.zero)
            new_loss = compute_loss(self.model, self.images, self.labels)
            delta    = new_loss - self.current_f  # negative means improvement

            tabu = self.is_tabu(weight_idx, bit_idx)
            asp  = (self.current_f + delta) < self.best_f  # beats global best

            if not tabu or asp:
                if delta < best_delta:
                    best_delta = delta
                    best_move  = (weight_idx, bit_idx)
                    if tabu and asp:
                        self.stats['aspiration_count'] += 1

        # Restore original weights before returning
        flat_int_to_weights(self.flat_int, self.model, self.scale, self.zero)
        return best_move, best_delta

    #escape random bits flip
    def escape_mechanism(self):
        self.stats['escape_count'] += 1
        steps = max(1, int(1 + (1 + random.random()) * self.moving_avg / 2))
        for _ in range(steps):
            move_id= random.randint(0, self.total_moves - 1)
            w, b = self.decode_move(move_id)
            self.flat_int = flip_bit(self.flat_int, w, b, self.bits)
            self.make_tabu(w, b)
            self.current_time += 1
        flat_int_to_weights(self.flat_int, self.model, self.scale, self.zero)
        self.current_f = compute_loss(self.model, self.images, self.labels)
        if self.current_f < self.best_f:
            self.best_f= self.current_f
            self.best_int= self.flat_int.copy()

    #main loop
    def search(self):
        start = time.time()
        print("\nStarting RTS.\n")

        while self.current_time < self.max_iterations:

            if self.check_for_repetition():
                self.escape_mechanism()
                self.current_time += 1
                continue

            best_move, best_delta = self.choose_best_move(n_candidates=50)

            if best_move is not None:
                weight_idx, bit_idx = best_move
                self.make_tabu(weight_idx, bit_idx)
                self.flat_int  = flip_bit(self.flat_int, weight_idx, bit_idx, self.bits)
                self.current_f += best_delta
                self.stats['loss_history'].append(self.current_f)

                if self.current_f < self.best_f:
                    self.best_f   = self.current_f
                    self.best_int = self.flat_int.copy()

            self.current_time += 1

            if self.current_time % 500 == 0:
                print(f"  iter {self.current_time:5d} | loss {self.current_f:.4f} "
                      f"| best {self.best_f:.4f} "
                      f"| list_size {self.list_size:.1f} "
                      f"| escapes {self.stats['escape_count']}")

        elapsed = time.time() - start

        # Load best weights found
        flat_int_to_weights(self.best_int, self.model, self.scale, self.zero)
        final_acc = compute_accuracy(self.model, self.images, self.labels)


        print("RESULTS")
        print(f"Best loss            : {self.best_f:.4f}")
        print(f"Final accuracy       : {final_acc:.3f}")
        print(f"Total iterations     : {self.current_time}")
        print(f"Escape count         : {self.stats['escape_count']}")
        print(f"Cycle detections     : {self.stats['cycle_detections']}")
        print(f"Aspiration count     : {self.stats['aspiration_count']}")
        print(f"Unique configs seen  : {len(self.pointer)}")
        bits_needed = math.ceil(math.log2(len(self.pointer))) if len(self.pointer) > 1 else 1
        print(f"Bits for hashing     : {bits_needed}")
        print(f"Time taken           : {elapsed:.1f}s")


if __name__ == "__main__":
    print("Loading MNIST subset")
    images, labels = load_mnist_subset(N_SAMPLES)

    print("Building network !")
    model = SmallNet()

    rts = RTS_NN(model, images, labels, max_iterations=MAX_ITERATIONS)
    rts.search()
  
#save the model 
save_dir = "/home/poornima/A14IV/models"
os.makedirs(save_dir, exist_ok=True)

# Save with timestamp and metadata
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = os.path.join(save_dir, f"tabu_nn_{timestamp}.pt")

torch.save({
    'model_state_dict': model.state_dict(),
    'quantization': {
        'scale': rts.scale,
        'zero': rts.zero,
        'bits': BITS_PER_WEIGHT
    },
    'hyperparams': {
        'best_loss': rts.best_f,
        'iterations': rts.current_time,
        'architecture': 'SmallNet'
    },
    'loss_history': rts.stats['loss_history']
}, checkpoint_path)

print(f" Model saved to: {checkpoint_path}")