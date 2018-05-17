import random, sys
from math import log, exp, factorial
from logarithms import log_add, log_add_list

### version 2, modified by Christian Schaffner on Saturday, 3 December 2016

class GeometricDistribution(object):
    '''An implementation of the geometric distribution whose support includes 0.'''

    def __init__(self, param=0.5):
        '''Constructor

        :param param: The parameter of this geometric distribution
        :raises: ValueError if param is not in [0,1]
        '''

        self.failure = log(1 - param)
        self.success = log(param)

    def log_prob(self, x):
        '''Compute the log-probability of an observation

        :param x: The observation
        :return: The log-probability of x under this distribution
        :raises: ValueError if x is not a non-negative integer
        '''

        if x%1 != 0 or x < 0:
            raise ValueError("x is not a non-negative integer")

        return self.success + x*self.failure

    def get_param(self):
        '''Get the parameter of this distribution

        :return: The parameter of this distribution
        '''
        return exp(self.success)


class EM(object):
    '''A geometric mixture model that can be trained using EM'''

    def __init__(self, num_components = 5, initial_mixture_weights = None, initial_geometric_parameters = None):
        '''Constructor

        :param num_components: The number of mixture components in this model
        :param initial_mixture_weights: A list of initial mixture weights (weights are initialised randomly if no list is provided)
        :param initial_geometric_parameters: A list of initial component parameters (initialised randomly if no list is provided)
        '''

        # total number of mixture components
        self.num_components = num_components
        # map from components to their mixture_weights (stored as logarithms)
        self.mixture_weights = list()
        # map from components to their distributions
        self.component_distributions = list()
        # map from components to their expected number of occurrence (stored as logarithms)
        self.expected_component_counts = dict()
        # map from components to expected values of observations (stored as logarithms)
        self.expected_observation_counts = dict()
        self.log_likelihood = 0
        self.initialise(initial_mixture_weights, initial_geometric_parameters)

    def initialise(self, initial_mixture_weights = None, initial_geometric_parameters = None):
        '''Initialise the parameters of this model

        :param initial_mixture_weights: A list of initial mixture weights (weights are initialised randomly if no list is provided)
        :param initial_geometric_parameters: A list of initial component parameters (initialised randomly if no list is provided)
        '''

        #TODO: the following code performs a random initialization of mixture weights and geometric distributions
        #TODO: make sure that the input arguments initial_mixtures_weights and initial_geometric_parameters
        #TODO: are used if they are actually provided.

        for component in range(self.num_components):
            mixture_weight = initial_mixture_weights[component] if initial_mixture_weights else random.random()
            param = initial_geometric_parameters[component] if initial_geometric_parameters else random.random()
            self.mixture_weights.append(mixture_weight)
            self.component_distributions.append(GeometricDistribution(param))
            # values that are stored as logarithms are initialized as -inf = log(0)
            self.expected_component_counts[component] = -float("inf")
            self.expected_observation_counts[component] = -float("inf")

        # normalise priors
        prior_sum = sum(self.mixture_weights)
        for component in range(self.num_components):
            # normalise and tranform to log-prob
            self.mixture_weights[component] = log(self.mixture_weights[component] / prior_sum)

 
    def train(self, data_path, iterations):
        '''Train the model on data for a fixed number of iterations. After each iteration of training, the log-likelihood,
        mixture weights and component parameters are printed out.

        :param data_path: The path to the data file
        :param iterations: The number of iterations
        '''

        for i in range(iterations):
            params = list()
            priors = list()
            for component in range(self.num_components):
                params.append(self.component_distributions[component].get_param())
                priors.append(exp(self.mixture_weights[component]))

            print("\nIteration {}".format(i))
            print("log-likelihood: {}".format(self.log_likelihood))
            print("Component priors: {}".format(priors))
            print("Component parameters: {}".format(params))

            # reset log-likelihood
            self.log_likelihood = 0

            self.em(data_path)


    def em(self, data_path):
        '''Perform one iteration of EM on the data

        :param data_path: The path to the data file
        '''
        with open(data_path) as data:
            for line in data:
                for observation in line.split():
                    self.e_step(int(observation))

        self.m_step()


    def e_step(self, observation):
        '''Perform the E-step on a single obervation. That is, compute the posterior of mixture components
        and add the expected occurrence of each component to the running totals
        self.log_likelihood ,
        self.expected_component_counts , and
        self.expected_observation_counts

        :param observation: The observation
        '''
        
        # Compute the joint probabilities p(x, c)
        joint_logprobs = list()
        for component in range(self.num_components):
            comp_dist = self.component_distributions[component]
            logprob = self.mixture_weights[component] + comp_dist.log_prob(observation)
            joint_logprobs.append(logprob)

        # Marginal probability p(x) = p(x, c_1) + ... + p(x, c_n)
        marginal_logprob = log_add_list(joint_logprobs)

        # Update log likelihood: log p(x_1, x_2, ...) = log p(x_1) + log p(x_2) + ...
        self.log_likelihood += marginal_logprob

        # Responsibilities are the posteriors p(c | x) = p(x, c) / p(x)
        # So log p(c | x) = log p(x, c) - log p(x)
        logresponsibilities = list()
        for component in range(self.num_components):
            dist = self.component_distributions[component]
            logresp = joint_logprobs[component] - marginal_logprob
            logresponsibilities.append(logresp)

        # Update the expected component count and expected observation counts
        for component in range(self.num_components):

            # log( expected observation counts )
            logresp = logresponsibilities[component]
            self.expected_component_counts[component] = log_add(self.expected_component_counts[component], logresp)

            # Compute the log( expected observation counts)
            # log (exp obs count) 
            #   = log( exp_obs_count + val * exp(logresp))
            #   = log( exp_obs_count + exp( log(val) + logresp ) )
            #   = log_add( log(exp_obs_count),  log(val) + logresp )
            if observation > 0:
                exp_obs = self.expected_observation_counts[component]
                self.expected_observation_counts[component] = log_add(exp_obs, log(observation) + logresp)

    def m_step(self):
        '''Perform the M-step. This step updates
        self.mixture_weights
        self.component_distributions
        '''

        # test if the sum of the summed expected_component_counts is roughly equal to the total amount of observations
        sum_obs_counts = log_add_list(self.expected_component_counts.values())
        print("Sum of expected component counts: {}".format(exp(sum_obs_counts)))


        # TODO: Implement this.
        # TODO: Make sure to reset the data structures you use for counting after you have
        # TODO: updated all parameters, namely self.expected_component_counts and self.expected_observation_counts

        for component in range(self.num_components):

            # log_new_weight 
            #    = log( exp( log_exp_count ) / exp( sum_obs_counts ) ) )
            #    = log( exp( log_exp_count )) - log( exp(sum_obs_counts) )
            #    = log_exp_count - sum_obs_counts
            log_exp_count = self.expected_component_counts[component]
            self.mixture_weights[component] = log_exp_count - sum_obs_counts
            
            # log_new_param
            #    = log( exp( log_exp_count) / (exp(log_exp_count) + exp(log_exp_sum)))
            #    = log_exp_count - (log( exp(log_exp_count) + exp(log_exp_sum) )
            #    = log_exp_count - log_add( log_exp_count + log_exp_sum )
            log_exp_sum = self.expected_observation_counts[component]
            logparam = log_exp_count - log_add(log_exp_count, log_exp_sum)
            self.component_distributions[component] = GeometricDistribution(exp(logparam))

        # Reset
        for component in range(self.num_components):
            self.expected_component_counts[component] = -float("inf")
            self.expected_observation_counts[component] = -float("inf")


def main(data_path, number_mixture_components, initial_mixture_weights=None, initial_geometric_parameters=None):
    learner = EM(int(number_mixture_components), initial_mixture_weights, initial_geometric_parameters)
    learner.train(data_path, 20)

if __name__ == "__main__":

    # x = 4
    # y = 5
    # logx = log(x)
    # logy = log(y)

    # print(exp(log_add(logx, logy)))

    # TODO: make sure that the first argument points to the data that you have downloaded
    # main('geometric_example_data.txt', 2, [0.2, 0.8], [0.2, 0.6])

    # TODO: use the following call instead when the small example above is running properly
    main('geometric_data.txt', 3, [.3, .3, .4], [.2, .5, .7])
