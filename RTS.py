import random
import math
import sys
import time
from basic_tabu import move_change , is_tabu, aspiration, make_tabu

n = 25
current_perm = random.sample(range(n), n)
flow = [[random.random() for _ in range(n)] for _ in range(n)]
distance = [[random.random() for _ in range(n)] for _ in range(n)]


class RTS:
    def __init__(self, flow, distance, n, max_iterations=10000):
        self.n = n
        self.flow = flow
        self.distance = distance
        self.max_iterations = max_iterations
        self.current_perm = random.sample(range(n), n)
        self.best_so_far = None
        self.current_f = self.total_cost_calculation(self.current_perm)
        self.best_perm = None
        self.current_time = 0
        self.list_size = 1
        self.lastest_occupation = [[float('-inf') for r in range(n)] for s in range(n)]
        self.chaotic = 0
        self.moving_average = self.n / 2
        self.steps_since_last_size_change = 0
        self.pointer = {} #dic to store the last time a configuration was seen and how many times it has been repeated

        self.statistics = {
            'escape_count': 0,
            'cycle_detections': 0,
            'aspiration_count': 0,
            'tenure_changes': [],
            'cost_history': [],
            'start_time': 0,
            'end_time': 0
        }

    def total_cost_calculation(self, perm): #calculating the total cost
        cost = 0
        for i in range(self.n):
            for j in range(self.n):
                cost += self.flow[i][j] * self.distance[perm[i]][perm[j]]
        return cost

    def initialization(self):
        self.current_perm = random.sample(range(self.n), self.n)
        self.current_f = self.total_cost_calculation(self.current_perm)
        self.best_so_far = self.current_f
        self.best_perm = self.current_perm.copy()
        self.current_time = 0
        config_tuple = tuple(self.current_perm)
        self.pointer[config_tuple] = {'last_time': self.current_time, 'repetitions': 0}
        self.statistics['cost_history'].append(self.current_f)
        self.statistics['start_time'] = time.time()
        print(f"Initial cost: {self.current_f}")
        return self.current_f

    def check_for_repetition(self, cycle_max=50, Rep=3, Chaos=3, Increase=1.1, Decrease=0.9):
        """Check for repetition and adjust tabu list size accordingly."""
        self.steps_since_last_size_change += 1
        configuration = tuple(self.current_perm)
        #if config found update 
        if configuration in self.pointer:
            length = self.current_time - self.pointer[configuration]['last_time']
            self.pointer[configuration]['last_time'] = self.current_time
            self.pointer[configuration]['repetitions'] += 1
           #print(f"Configuration {configuration} found again. Repetitions: {self.pointer[configuration]['repetitions']}")  
            if self.pointer[configuration]['repetitions'] > Rep:
                self.chaotic += 1
                self.statistics['cycle_detections'] += 1
                if self.chaotic > Chaos:
                    self.chaotic = 0
                    return True  # escape

            if length < cycle_max:
                self.moving_average = 0.1 * length + 0.9 * self.moving_average
                # Only increase list size on short cycles, not every visit
                self.list_size = self.list_size * Increase
                self.steps_since_last_size_change = 0
        else:
            self.pointer[configuration] = {'last_time': self.current_time, 'repetitions': 0}

        if self.steps_since_last_size_change > self.moving_average:
            self.list_size = max(self.list_size * Decrease, 1)
            self.steps_since_last_size_change = 0

        return False  # No escape needed

    def reactive_tabu_search(self, max_iterations=10000, target=None):
        """Main RTS loop"""
        while self.current_time < max_iterations:

            escape = self.check_for_repetition()

            if escape:
                # Only escape when cycling is detected
                self.escape_mechanism()
                self.current_time += 1
                continue

            # Normal tabu move
            moves = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
            best_move, best_delta = self.choose_best_move(moves)

            if best_move is not None:
                r_chosen, s_chosen = best_move

                # Apply the move
                self.current_perm[r_chosen], self.current_perm[s_chosen] = (
                    self.current_perm[s_chosen], self.current_perm[r_chosen]
                )
                make_tabu(r_chosen, s_chosen, self.current_perm, self.lastest_occupation, self.current_time)

                #  Use delta for cost update
                self.current_f += best_delta
                self.statistics['cost_history'].append(self.current_f)

                # Update best solution found
                if self.current_f < self.best_so_far:
                    self.best_so_far = self.current_f
                    self.best_perm = self.current_perm.copy()

            self.current_time += 1

            if target is not None and self.best_so_far <= target:
                self.statistics['end_time'] = time.time()
                return "SUCCESSFUL"

        self.statistics['end_time'] = time.time()
        return "UNSUCCESSFUL"
    
    def choose_best_move(self, moves):
        """Select the best non tabu move, or tabu move if it passes aspiration."""
        best_move = None
        best_delta = float('inf')
        tenure = int(self.list_size)  # must be int, cant have partial interations
        for r, s in moves:
            delta = move_change(r, s, self.current_perm, self.flow, self.distance, self.n)
            tabu = is_tabu(r, s, self.current_perm, self.lastest_occupation, tenure, self.current_time)
            # Aspiration move if new global best 
            asp = aspiration(self.current_f, delta, self.best_so_far)
            if not tabu or asp:
                if delta < best_delta:
                    best_delta = delta
                    best_move = (r, s)
                    if tabu and asp:
                        self.statistics['aspiration_count'] += 1

        return best_move, best_delta

    def escape_mechanism(self):
        """Random walk to escape cycling / local optima"""
        self.statistics['escape_count'] += 1

        steps = int(1 + (1 + random.random()) * self.moving_average / 2)
        steps = max(1, steps)

        for _ in range(steps):
            r = random.randint(0, self.n - 1)
            s = random.randint(0, self.n - 1)
            while s == r:
                s = random.randint(0, self.n - 1)
            self.current_perm[r], self.current_perm[s] = self.current_perm[s], self.current_perm[r]
            make_tabu(r, s, self.current_perm, self.lastest_occupation, self.current_time)
            self.current_time += 1

        # Recalculate cost fully after escape 
        self.current_f = self.total_cost_calculation(self.current_perm)

        if self.current_f < self.best_so_far:
            self.best_so_far = self.current_f
            self.best_perm = self.current_perm.copy()
            #print("escape move made: ",self.current_f)

if __name__ == "__main__":
    n = 25
    flow = [[random.random() for _ in range(n)] for _ in range(n)]
    distance = [[random.random() for _ in range(n)] for _ in range(n)]

    rts = RTS(flow, distance, n, max_iterations=10000)
    rts.initialization()
    search_result = rts.reactive_tabu_search(max_iterations=10000, target=None)

    print("problem size (n):", n)
    print("Best solution found:", rts.best_so_far)
    print("Best permutation:", rts.best_perm)
    print("escape count:", rts.statistics['escape_count'])
    print("cycle detections:", rts.statistics['cycle_detections'])
    print("aspiration count:", rts.statistics['aspiration_count'])
    print("total search space size:", math.factorial(n))
    print("total iterations:", rts.current_time)
    print("time taken:", rts.statistics['end_time'] - rts.statistics['start_time'])
    print("size of pointer dict:", len(rts.pointer))
    print("no of bits for hashing:", math.ceil(math.log2(len(rts.pointer))) if rts.pointer else 0)


