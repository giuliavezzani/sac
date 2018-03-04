import abc
import gtimer as gt

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts


class RLAlgorithmMultiTask(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            batch_size=64,
            n_epochs=1000,
            n_train_repeat=1,
            epoch_length=1000,
            min_pool_size=10000,
            max_path_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
            num_tasks=2,
    ):
        """
        Args:
            batch_size (`int`): Size of the sample batch to be used
                for training.
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            min_pool_size (`int`): Minimum size of the sample pool before
                running training.
            max_path_length (`int`): Number of timesteps before resetting
                environment and policy, and the number of paths used for
                evaluation rollout.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._min_pool_size = min_pool_size
        self._max_path_length = max_path_length

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._num_tasks = num_tasks
        self._batch_size_per_task = int(self._batch_size / self._num_tasks)

        self._sess = tf_utils.get_default_session()

        self.envs = []
        self.policies = []
        self.pools = []


    def _train(self, envs, policies, pools):
        """Perform RL training.
        Args:
            env (`rllab.Env`): Environment used for training
                               ### Envs will be generated in _init_training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._init_training(envs, policies, pools)

        with self._sess.as_default():
            n_episodes = 0
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            observations = []
            path_lenghts = []
            path_returns = []
            last_path_returns = []
            max_path_returns = []
            terminals = []

            for task in range(self._num_tasks):
                observations.append(self.envs[task].reset(reset_args=task))
                self.policies[task].reset()

                path_lenghts.append(0)
                path_returns.append(0)
                last_path_returns.append(0)
                max_path_returns.append(-np.inf)
                terminals.append(0)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    iteration = t + epoch * self._epoch_length

                    for task in range(self._num_tasks):
                        #envs[task]._goal_idx = task
                        action, _ = self.policies[task].get_action(observations[task], task)
                        next_ob, reward, terminal, info = self.envs[task].step(action)
                        #print('task ', task)
                        #print('goal', self.envs[task].goals[envs[task]._goal_idx])
                        path_lenghts[task] += 1
                        path_returns[task] += reward
                        #print('rew', reward)

                        ### Different pool for each task
                        self.pools[task].add_sample(
                            observations[task],
                            action,
                            reward,
                            terminal,
                            next_ob,
                        )

                        if terminal or path_lenghts[task] >= self._max_path_length:
                            observations[task] = self.envs[task].reset(reset_args=task)
                            self.policies[task].reset()
                            path_lenghts[task] = 0
                            max_path_returns[task] = max(max_path_returns[task], path_returns[task])
                            last_path_returns[task] = path_returns[task]
                            terminals[task] = terminal

                            path_returns[task] = 0

                        else:
                            observations[task] = next_ob

                    if (terminals[task] == True or
                         path_lenghts[task] >= self._max_path_length
                         for task in range(self._num_tasks)):
                        n_episodes += 1

                    gt.stamp('sample')
                    for i in range(self._n_train_repeat):
                        batchs = []
                    ### BUG ALERT: to be handle the case where batch is not of the correct slide
                        for task in range(self._num_tasks):
                            if self.pools[task].size >= self._min_pool_size:
                                batchs.append(self.pools[task].random_batch(self._batch_size_per_task))

                        if len(batchs) == self._num_tasks:
                            batch = {}

                            for k in batchs[task].keys():
                                batch[k] = np.concatenate([batchs[task][k] for task in range(self._num_tasks)])
                            self._do_training(iteration, batch)

                    gt.stamp('train')

                print('eval')
                #self._evaluate(epoch)
                print('after eval')
                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('-------------------------------------------', '-')
                logger.record_tabular('Training', '-')
                logger.record_tabular('-------------------------------------------', '-')
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('episodes', n_episodes)
                for task in range(self._num_tasks):
                    logger.record_tabular('-------------------------------------------', '-')
                    logger.record_tabular('Task no. :', str(task))
                    logger.record_tabular('max-path-return-' + str(task), max_path_returns[task])
                    logger.record_tabular('last-path-return-' + str(task), last_path_returns[task])
                    logger.record_tabular('pool-size-' + str(task), self.pools[task].size)

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            env.terminate()

    def _evaluate(self, epoch):
        """Perform evaluation for the current policies.
        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        paths = []
        logger.record_tabular('Evaluation', '-')
        for task in range(self._num_tasks):
            print(task)
            paths.append(rollouts(self._eval_envs[task], self.policies[task], self._max_path_length,
                         self._eval_n_episodes, task))

            total_returns = [path['rewards'].sum() for path in paths[task]]
            episode_lengths = [len(p['rewards']) for p in paths[task]]
            logger.record_tabular('-------------------------------------------', '-')
            logger.record_tabular('Task no. :', str(task))
            logger.record_tabular('return-average-' + str(task), np.mean(total_returns))
            logger.record_tabular('return-min-' + str(task), np.min(total_returns))
            logger.record_tabular('return-max-' + str(task), np.max(total_returns))
            logger.record_tabular('return-std-' + str(task), np.std(total_returns))
            logger.record_tabular('episode-length-avg-' + str(task), np.mean(episode_lengths))
            logger.record_tabular('episode-length-min-' + str(task), np.min(episode_lengths))
            logger.record_tabular('episode-length-max-' + str(task), np.max(episode_lengths))
            logger.record_tabular('episode-length-std-' + str(task), np.std(episode_lengths))
            logger.record_tabular('epoch', epoch)

            self._eval_envs[task].log_diagnostics(paths[task])
            if self._eval_render:
                self._eval_envs[task].render(paths[task])

            batch = self.pools[task].random_batch(self._batch_size_per_task)

            #self.log_diagnostics(batch, task)

    @abc.abstractmethod
    def log_diagnostics(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, itr, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, envs, policies, pools):
        """Method to be called at the start of training.
        :param env: Environment instance.
        :param policy: Policy instance.
        :return: None
        """

        self.envs = envs
        self.policies = policies
        self.pools = pools

        self._eval_envs = []
        for i in range(self._num_tasks):
            if self._eval_n_episodes > 0:
                self._eval_envs.append(deep_clone(self.envs[i]))
                #self._eval_envs.append(self.envs[i])
#self._eval_envs.append(AntEnvRandGoalRing(self.file_goals,file_env=self.file_env, goal=i))
