from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import prettytensor as pt
import tempfile
import sys

dtype = tf.float32
eps = 1e-6
config={"max_steps": 1000, "episodes_per_roll": 1000, "gamma": 0.95, "cg_damping": 0.1, "max_kl": 0.01}

class TRPOAgent(object):

    def __init__(self, env):
        self.env = env
        self.session = tf.Session()

        self.state_dim = state_dim = env.observation_space.shape[0]
        self.action_dim = action_dim = env.action_space.n

        self.train = True
        self.end_count = 0

        self.prev_action = np.zeros((1, action_dim))
        self.state = state = tf.placeholder(dtype, shape=[None, state_dim], name="state")
        self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, action_dim], name="oldaction_dist")

        # Create NeuralNetwork
        self.action_dist, _ = action_dist, _ = (pt.wrap(self.state). # HERE, must we use "self.state" rather than "state"
            fully_connected(64, activation_fn=tf.nn.tanh).
            softmax_classifier(action_dim))

        N = tf.shape(state)[0] # SEEMS like that both the "self.state" and "state" are ok.

        p_n = slice_2d(action_dist, tf.range(0, N), action)
        oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
        ratio_n = p_n / oldp_n
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()
        kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + eps) / (action_dist + eps))) / Nf
        ent = tf.reduce_sum(-action_dist * tf.log(action_dist + eps)) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)

        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(action_dist) * tf.log(tf.stop_gradient(action_dist + eps) / (action_dist + eps))) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list) # map( fun, arg )

        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size

        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)

        self.vf = VF(self.session)
        self.session.run(tf.initialize_all_variables())    
    def act(self, state):
        state = np.expand_dims(state, 0)
        action_dist = self.session.run(self.action_dist, {self.state:state})
        if self.train:
            action = int(cat_sample(action_dist)[0])
        else:
            self.env.render()
            action = int(np.argmax(action_dist))

        self.prev_action *= 0.0
        self.prev_action[0, action] = 1.0
        return action, action_dist, np.squeeze([state]) # HERE "state" is useless
    def learn(self):
        start_time = time.time()
        i = 0
        numeptotal = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config["max_steps"],
                config["episodes_per_roll"])

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config["gamma"])
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist = np.concatenate([path["action_dists"] for path in paths])
            state = np.concatenate([path["obs"] for path in paths])
            action = np.concatenate([path["actions"] for path in paths])
            baseline = np.concatenate([path["baseline"] for path in paths])
            returns = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant = np.concatenate([path["advant"] for path in paths])
            advant -= advant.mean()
            advant /= (advant.std() + 1e-8)

            feed = {self.state: state,
                    self.action: action,
                    self.advant: advant,
                    self.oldaction_dist: action_dist}

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p # WHAT "self.flat_tangent"
                return self.session.run(self.fvp, feed) + config["cg_damping"] * p # WHAT 'config["cg_damping"]'
            def loss(th):
                self.sff(th)
                return self.session.run(self.losses[0], feed_dict=feed)

            episoderewards = np.array([path["rewards"].sum() for path in paths])

            print("\n********** Iteration %i ************" % i)

            if episoderewards.mean() > 1.1*500:
                self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                self.vf.fit(paths)
                thprev = self.gf()

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / config["max_kl"])
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config["max_kl"]:
                    self.sff(thprev)

                stats = {}
                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline), np.array(returns))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if entropy != entropy:
                    exit(-1)
                if exp > 0.8:
                    self.train = False
            i += 1


env = gym.make("CartPole-v0")
agent = TRPOAgent(env)
agent.learn()


