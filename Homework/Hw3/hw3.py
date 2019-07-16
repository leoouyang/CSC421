import numpy as np

def sigma(x):
    return int(x >= 0)

def rnn(x1, x2, u, v, w, bh, by):
    sigma_v = np.vectorize(sigma)

    output = ""
    prev_hidden_out = np.zeros(3)
    x1 = "0" + x1
    x2 = "0" + x2

    for i in range(len(x1)-1, -1, -1):
        input = np.array([int(x1[i]), int(x2[i])])
        h = w.dot(prev_hidden_out) + u.dot(input) + bh
        prev_hidden_out = sigma_v(h)
        result = sigma_v(v.dot(prev_hidden_out) + by)
        output = str(result) + output
    return output

u = np.array([[1,1],[1,1],[1,1]])
v = np.array([1,-1,2])
w = np.array([[0,1,0],[0,1,0],[0,1,0]])
bh = np.array([-0.5,-1.5,-2.5])
by = -0.5
