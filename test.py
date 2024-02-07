def kl_annealer(n_epoch, kl_start, kl_w_start, kl_w_end):
    i_start = kl_start
    w_start = kl_w_start
    w_max = kl_w_end

    inc = (w_max - w_start) / (n_epoch - i_start)

    annealing_weights = []
    for i in range(n_epoch):
        k = (i - i_start) if i >= i_start else 0
        annealing_weights.append(w_start + k * inc)

    return annealing_weights

# Example usage:
n_epoch = 100
kl_start = 10
kl_w_start = 0.1
kl_w_end = 0.5

annealing_weights = kl_annealer(n_epoch, kl_start, kl_w_start, kl_w_end)
print(annealing_weights)


class KLAnnealer:
    def __init__(self, n_epoch):
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc
    
kl_annealer = KLAnnealer(n_epoch)
for i in range(n_epoch):
    print(kl_annealer(i) == annealing_weights[i])