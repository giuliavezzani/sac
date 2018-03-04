import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from sac.misc.mlp import mlp
from sac.misc.mlp import mlp_extra
from sac.misc import tf_utils


class ValueFunction(Parameterized, Serializable):

    def __init__(self, name, input_pls, hidden_layer_sizes, hidden_layer_sizes_extra, task):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._input_pls = input_pls
        if not hidden_layer_sizes==None:
            self._layer_sizes = list(hidden_layer_sizes) + [None]
        else:
            self._layer_sizes = None

        self._layer_sizes_extra  = list(hidden_layer_sizes_extra) + [None]

        self._output_t = []
        for task in range(self._num_tasks):
            self._output_t.append(self.get_output_for(*self._input_pls, task=task, reuse=tf.AUTO_REUSE))

        print(self.get_params_internal())
            #self._output_t.append(self.get_output_for(*self._input_pls, task=task))

    def get_output_for(self, *inputs, task, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            if not self._layer_sizes == None:
                value_t = mlp(
                    inputs=inputs,
                    output_nonlinearity=None,
                    layer_sizes=self._layer_sizes,
                )  # N

        original_name = self._name
        print('name', self._name)
        self._name = self._name + '/task-' + str(task)

        with tf.variable_scope(self._name, reuse=reuse):
            if not self._layer_sizes == None:
                value_t = mlp_extra(
                    inputs=value_t,
                    output_nonlinearity=None,
                    layer_sizes=self._layer_sizes,
                    layer_sizes_extra=self._layer_sizes_extra,
                )
            else:
                value_t = mlp(
                    inputs=inputs,
                    output_nonlinearity=None,
                    layer_sizes=self._layer_sizes_extra,
                )

        #import IPython
        #IPython.embed()
        self._name = original_name
        return value_t

    def eval(self, *inputs, task):
        self._task = task
        feeds = {pl: val for pl, val in zip(self._input_pls, inputs)}

        return tf_utils.get_default_session().run(self._output_t[task], feeds)

    def get_params_internal(self,scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope = tf.get_variable_scope().name


        scope += '/' + self._name + '/' if len(scope) else self._name + '/'

        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )


class NNVFunctionMultiHead(ValueFunction):

    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=None,
                 hidden_layer_sizes_extra=None,
                 name='value_function',
                 task=0,
                 num_tasks=2):
        Serializable.quick_init(self, locals())

        self._num_tasks = num_tasks
        self._Do = env_spec.observation_space.flat_dim
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        super(NNVFunctionMultiHead, self).__init__(
            'value_function', (self._obs_pl,), hidden_layer_sizes=hidden_layer_sizes,
             hidden_layer_sizes_extra=hidden_layer_sizes_extra, task=task)


class NNQFunctionMultiHead(ValueFunction):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=None,
                 hidden_layer_sizes_extra=None,
                 name='q_function',
                 task=0,
                 num_tasks=2):
        Serializable.quick_init(self, locals())

        self._num_tasks = num_tasks
        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        super(NNQFunctionMultiHead, self).__init__(
            'qf', (self._obs_pl, self._action_pl), hidden_layer_sizes, hidden_layer_sizes_extra, num_tasks)
