import autograd.numpy as np
import autograd as ag
from autograd.misc import flatten
from matplotlib import pyplot as plt


def relu(z):
    return np.maximum(z, 0.)

def net_predict(params, x):
    """Compute the output of a ReLU MLP with 2 hidden layers."""
    H1 = relu(np.outer(x, params['W1']) + params['b1'])
    H2 = relu(np.dot(H1, params['W2']) + params['b2'])
    return np.dot(H2, params['w3']) + params['b3']

def random_init(std, nhid):
    return {'W1': np.random.normal(0, std, size=nhid),
            'b1': np.random.normal(0., std, size=nhid),
            'W2': np.random.normal(0., std, size=(nhid,nhid)),
            'b2': np.random.normal(0., std, size=nhid),
            'w3': np.random.normal(0., std, size=nhid),
            'b3': np.random.normal(0., std)}

class ToyDataGen:
    """Samples a random piecewise linear function, and then samples noisy
    observations of the function."""
    def __init__(self, xmin, xmax, ymin, ymax, std, num_pieces):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.std = std
        self.num_pieces = num_pieces

    def sample_dataset(self, npts):
        x = np.random.uniform(self.xmin, self.xmax, size=npts)
        heights = np.random.uniform(self.ymin, self.ymax, size=self.num_pieces)
        bins = np.floor((x - self.xmin) / (self.xmax - self.xmin) * self.num_pieces).astype(int)
        y = np.random.normal(heights[bins], self.std)
        return x, y

def gd_step(cost, params, lrate):
    """Perform one gradient descent step on the given cost function with learning
    rate lrate. Returns a new set of parameters, and (IMPORTANT) does not modify
    the input parameters."""
    ### YOUR CODE HERE

    cost_grad = ag.grad(cost)
    param_grads = cost_grad(params)
    flat_params, unflatten_func = flatten(params)
    flat_grads, unflatten_func2 =flatten(param_grads)
    return unflatten_func(flat_params - flat_grads * lrate)
    ### END CODE


class InnerObjective:
    """Mean squared error."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, params):
        return 0.5 * np.mean((self.y - net_predict(params, self.x)) ** 2)

class MetaObjective:
    """Mean squared error after some number of gradient descent steps
    on the inner objective."""
    def __init__(self, x, y, inner_lrate, num_steps):
        self.x = x
        self.y = y
        self.inner_lrate = inner_lrate
        self.num_steps = num_steps

    def __call__(self, params, return_traj=False):
        """Compute the meta-objective. If return_traj is True, you should return
        a list of the parameters after each update. (This is used for visualization.)"""
        trajectory = [params]
        ### YOUR CODE HERE
        inner = InnerObjective(self.x, self.y)
        new_params = params
        for i in range(self.num_steps):
            new_params = gd_step(inner.__call__, new_params, self.inner_lrate)
            if return_traj:
                trajectory.append(new_params)

        final_cost = inner(new_params)
        ### END CODE

        if return_traj:
            return final_cost, trajectory
        else:
            return final_cost

    def visualize(self, params, title, ax):
        _, trajectory = self(params, return_traj=True)

        ax.plot(self.x, self.y, 'bx', ms=3.)
        px = np.linspace(XMIN, XMAX, 1000)
        for i, new_params in enumerate(trajectory):
            py = net_predict(new_params, px)
            ax.plot(px, py, 'r-', alpha=(i+1)/len(trajectory))
        ax.set_title(title)


OUTER_LRATE = 0.01
OUTER_STEPS = 12000
INNER_LRATE = 0.1
INNER_STEPS = 5

PRINT_EVERY = 100
DISPLAY_EVERY = 1000

XMIN = -3
XMAX = 3
YMIN = -3
YMAX = 3
NOISE = 0.1
BINS = 6
NDATA = 100

INIT_STD = 0.1
NHID = 50

def train():
    np.random.seed(0)
    data_gen = ToyDataGen(XMIN, XMAX, YMIN, YMAX, NOISE, BINS)
    params = random_init(INIT_STD, NHID)
    fig, ax = plt.subplots(3, 4, figsize=(16, 9))
    plot_id = 0

    x_val, y_val = data_gen.sample_dataset(NDATA)

    for i in range(OUTER_STEPS):
        ### YOUR CODE HERE
        x,y = data_gen.sample_dataset(NDATA)
        cost = MetaObjective(x, y, INNER_LRATE,INNER_STEPS)
        params = gd_step(cost.__call__, params, OUTER_LRATE)
        ### END CODE

        if (i+1) % PRINT_EVERY == 0:
            val_cost = MetaObjective(x_val, y_val, INNER_LRATE, INNER_STEPS)
            print('Iteration %d Meta-objective: %1.3f' % (i+1, val_cost(params)))

        #print('Outer cost:', cost(params))
        if (i+1) % DISPLAY_EVERY == 0:
            cost.visualize(params, 'Iteration %d' % (i+1), ax.flat[plot_id])
            plot_id += 1

if __name__ == "__main__":
	train()
	plt.show()
