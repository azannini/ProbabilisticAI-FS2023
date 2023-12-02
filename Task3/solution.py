"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern
from scipy.stats import norm
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.logP = GaussianProcessRegressor(kernel=0.2*Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5) + WhiteKernel(0.15))
        self.SA = GaussianProcessRegressor(kernel=(4 + np.sqrt(2)*Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5)+ DotProduct()) + WhiteKernel(0.0001))
        self.X = np.array([])
        self.F = np.array([])
        self.V = np.array([])

        self.af_lambda = 5.5
        self.beta = 0.01
        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        if len(self.X) == 0:
            return get_initial_safe_point()
        else:
            x_opt = self.optimize_acquisition_function()
            return x_opt

        # raise NotImplementedError

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt
    

    def UBC_f(self, x: np.ndarray):

        mu = self.logP.predict(x)[0]
        sigma = self.logP.predict(x,return_std=True)[1]

        return mu + self.beta * sigma
    
    def UBC_v(self, x: np.ndarray):

        mu = self.SA.predict(x)[0]
        sigma = self.SA.predict(x,return_std=True)[1]

        return mu + self.beta * sigma

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """

        type = 'EI'
        x = np.atleast_2d(x)
        
        if type == 'UBC':

            #Upper Confidence Bound (UCB)
           

            af = self.UBC_f(x) - self.af_lambda * max(self.UBC_v(x),0)

        elif type == 'EI':

            #Expected Improvement (EI)
            mu = self.logP.predict(x)[0]
            sigma = self.logP.predict(x,return_std=True)[1]

            mu_v = self.SA.predict(x)[0]
            sigma_v = self.SA.predict(x,return_std=True)[1]

            f_cap = max(self.F)
            z = (mu - f_cap)/sigma

            af = (mu - f_cap) * norm.cdf(z) + sigma * norm.pdf(z) * norm.cdf((SAFETY_THRESHOLD - mu_v)/sigma_v)**2 - self.af_lambda * max(mu_v - SAFETY_THRESHOLD,0)


        return af


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        # print('the data point is: ','data: ', x,'value of f: ', f,'value of v: ',v)

        self.X = np.append(self.X, x)
        self.F = np.append(self.F, f)
        self.V = np.append(self.V, v)

        self.logP.fit(self.X.reshape(-1,1), self.F.reshape(-1,1))
        self.SA.fit(self.X.reshape(-1,1), self.V.reshape(-1,1))

        

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # raise NotImplementedError
        max = 0

        resolution = 0.001

        for i in np.arange(DOMAIN[0][0], DOMAIN[0][1], resolution):
            mu_f , sigma_f = self.logP.predict(np.array([i]).reshape(-1,1), return_std=True)
            mu_v , sigma_v = self.SA.predict(np.array([i]).reshape(-1,1), return_std=True)

            if norm.cdf((SAFETY_THRESHOLD - mu_v)/sigma_v) > 0.99 :
             if mu_f > max:
                max = mu_f
                x_opt = i

        return x_opt

            

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    
    """Dummy SA constraint"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()