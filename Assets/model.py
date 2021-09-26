from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Lambda, Concatenate, Add, GaussianNoise, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from CustomLayer import NoisyDense, LayerNormalization

from collections import namedtuple, deque
from functools import partial
import numpy as np
import math
import itertools


class DDPG():
    def __init__(
                self, 
                dim_states, 
                dim_actions,
                range_action_high,
                range_action_low,
                hidden_layers_a,
                hidden_layers_c,
                kernel_initializer,
                l2_reg_a,
                l2_reg_c,
                BN_a,
                BN_c,
                activ_a,
                optim_a,
                optim_c,
                lr_a,
                lr_c,
                tau,
                perturbation,
                theta,
                sigma_init,
                sigma,
                # hidden_layers_f,
                # lr_f,
                TD3,
                # icm,
                ):
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.range_action_high= range_action_high.reshape((1,-1))
        self.range_action_low= range_action_low.reshape((1,-1))
        self.range_action_range = (range_action_high - range_action_low) / 2.
        self.range_action_mean = (range_action_high + range_action_low) / 2.

        if type(hidden_layers_a) is int:
            self.hidden_layers_a = self.HiddenLayersSelector(hidden_layers_a)
            self.hidden_layers_c = self.HiddenLayersSelector(hidden_layers_c)
        elif type(hidden_layers_a) is list:
            self.hidden_layers_a = self.HiddenLayersCreator(hidden_layers_a)
            self.hidden_layers_c = self.HiddenLayersCreator(hidden_layers_c)
        self.kernel_initializer = kernel_initializer
        self.l2_reg_a = l2_reg_a
        self.l2_reg_c = l2_reg_c
        self.BN_a = bin(BN_a)[3:]
        self.BN_c = bin(BN_c)[3:]
        self.activ_a = activ_a
        self.lr_a = lr_a
        self.lr_c = lr_c
        # self.lr_f = lr_f
        self.optim_a = Adam(lr=self.lr_a) if optim_a == 'Adam' else SGD(lr=self.lr_a)
        self.optim_c = Adam(lr=self.lr_c) if optim_c == 'Adam' else SGD(lr=self.lr_c)
        self.TD3 = TD3
        # self.icm = icm

        self.tau = tau # 0.001
        self.gamma = 0.99

        self.perturbation = perturbation
        self.mu = 0.0
        self.theta = theta
        self.sigma_init = sigma_init
        self.sigma = sigma
        self.dt = 0.0165 / 4. # bullet3/examples/pybullet/gym/pybullet_envs/gym_locomotion_envs.py
        self.ou_noise = None
        self.paramas_noise_sigma = 1.0
        self.prev_weights_dict = {}

        # if type(hidden_layers_f) is int:
        #     self.hidden_layers_f = self.HiddenLayersSelector(hidden_layers_f)
        # elif type(hidden_layers_f) is list:
        #     self.hidden_layers_f = self.HiddenLayersCreator(hidden_layers_f)
        
        #-------------------------------
        # Compile CriticNetwork
        #-------------------------------    
        self.critic_network = self.build_critic_network()

        input_state = Input(shape=(self.dim_states,))
        input_action = Input(shape=(self.dim_actions,))
        # input_weight = Input(shape=(1,))

        output_critic_layer = self.critic_network([input_state, input_action])

        # weighted_mse = partial(self.weighted_mse, input_weight=input_weight)
        # weighted_mse = self.get_weighted_mse(input_weight=input_weight)       
        # weighted_mse.__name__ = 'weighted_mse'
        # get_custom_objects().update({'weighted_mse': weighted_mse})
        get_custom_objects().update({'merged_weighted_mse': self.merged_weighted_mse})

        # self.critic_network_model = Model(inputs = [input_state, input_action, input_weight],
        #                                   outputs = output_critic_layer)

        self.critic_network_model = Model(inputs = [input_state, input_action],
                                          outputs = output_critic_layer)
        # print('CriticNetwork model Summary:')
        print('CriticNetwork Summary', end='---Hidden Layers:')
        print(self.hidden_layers_c)
        self.critic_network_model.summary()
        self.critic_network_model.compile(optimizer=self.optim_c,
                                          loss=self.merged_weighted_mse)

        #-------------------------------
        # Compile ActorNetwork
        #-------------------------------
        #self.critic_network.trainable = False
        self.actor_network = self.build_actor_network()

        input_state = Input(shape=(self.dim_states,))
        output_actor_layer = self.actor_network(input_state)

        # neg_J = partial(self.neg_J, input_state=input_state)
        # neg_J = self.get_neg_J(input_state=input_state)
        # neg_J.__name__ = 'neg_J'
        # get_custom_objects().update({'neg_J': neg_J})
        get_custom_objects().update({'merged_neg_J': self.merged_neg_J})

        self.actor_network_model = Model(inputs = [input_state],
                                         outputs = output_actor_layer)
        # print('ActorNetwork model Summary:')
        print('ActorNetwork Summary', end='---Hidden Layers:')
        print(self.hidden_layers_a)
        self.actor_network_model.summary()
        self.actor_network_model.compile(optimizer=self.optim_a,
                                         loss=self.merged_neg_J)

        #-------------------------------
        # Compile TargetNetworks
        #-------------------------------
        self.target_actor_network = self.build_actor_network()
        self.target_critic_network = self.build_critic_network()

        #-------------------------------
        # Compile Critic2Network & TargetNetwork for TD3
        #-------------------------------    
        if self.TD3:
            self.critic_2_network = self.build_critic_network()

            input_state = Input(shape=(self.dim_states,))
            input_action = Input(shape=(self.dim_actions,))
            # input_weight = Input(shape=(1,))

            output_critic_2_layer = self.critic_2_network([input_state, input_action])

            # weighted_mse = partial(self.weighted_mse, input_weight=input_weight)
            # weighted_mse = self.get_weighted_mse(input_weight=input_weight)       
            # weighted_mse.__name__ = 'weighted_mse'
            # get_custom_objects().update({'weighted_mse': weighted_mse})
            get_custom_objects().update({'merged_weighted_mse': self.merged_weighted_mse})

            # self.critic_2_network_model = Model(inputs = [input_state, input_action, input_weight],
            #                                   outputs = output_critic_layer)

            self.critic_2_network_model = Model(inputs = [input_state, input_action],
                                                outputs = output_critic_2_layer)
            # print('CriticNetwork model Summary:')
            print('Critic2Network for TD3 Summary', end='---Hidden Layers:')
            print(self.hidden_layers_c)
            self.critic_2_network_model.summary()
            self.critic_2_network_model.compile(optimizer=self.optim_c,
                                                loss=self.merged_weighted_mse)

            self.target_critic_2_network = self.build_critic_network()

        # # -------------------------------
        # # Compile Forward model from ICM
        # # -------------------------------
        # if self.icm:
        #     self.forward = self.build_forward()

        #     input_state = Input(shape=(self.dim_states,))
        #     input_action = Input(shape=(self.dim_actions,))
        #     next_state_pred = self.forward([input_state, input_action])
        #     self.forward_model = Model(inputs = [input_state, input_action], 
        #                             outputs = next_state_pred)
        #     # print('Forward model Summary:')
        #     # self.forward_model.summary()
        #     self.forward_model.compile(optimizer=Adam(lr=self.lr_f),
        #                             loss='mean_squared_error')  


    # def get_neg_J(self, input_state):
    #     def neg_J(y_true, y_pred):
    #         dQda = K.gradients(self.critic_network([input_state, y_pred]), y_pred)
    #         return - y_pred * dQda # y_pred = a
    #     return neg_J


    # def get_weighted_mse(self, input_weight):
    #     def weighted_mse(y_true, y_pred):
    #         return K.mean(K.square(y_pred - y_true) * input_weight, axis=-1)
    #     return weighted_mse


    # def neg_J(self, y_true, y_pred, input_state):
    #     dQda = K.gradients(self.critic_network([input_state, y_pred]), y_pred)
    #     return - y_pred * dQda # y_pred = a


    # def weighted_mse(self, y_true, y_pred, input_weight):
    #     return K.mean(K.square(y_pred - y_true) * input_weight, axis=-1)


    def merged_neg_J(self, y_true, y_pred):
        input_state = K.reshape(y_true, (-1, self.dim_states))
        dQda = K.gradients(self.critic_network([input_state, y_pred]), y_pred)
        return - y_pred * dQda # y_pred = a


    def merged_weighted_mse(self, y_true, y_pred):
        input_weight = K.reshape(y_true, (-1, 2))[:,1]
        y_true = K.reshape(y_true, (-1, 2))[:,0]
        input_weight = K.reshape(input_weight, (-1, 1))
        y_true = K.reshape(y_true, (-1, 1))
        return K.mean(K.square(y_pred - y_true) * input_weight, axis=-1)


    def build_actor_network(self): # Action
        input_layer = Input(shape=(self.dim_states,))
        x = NoisyDense(units=self.hidden_layers_a[0], sigma_init=self.sigma_init)(input_layer) if self.perturbation in [3, 4] else Dense(units=self.hidden_layers_a[0], kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(l=self.l2_reg_a))(input_layer)
        if int(self.BN_a[0]): x = BatchNormalization()(x)
        if self.perturbation == 2: x = LayerNormalization()(x)
        x = Activation(self.activ_a[0])(x)
        for i in range(1, len(self.hidden_layers_a)):
            x = NoisyDense(units=self.hidden_layers_a[i], sigma_init=self.sigma_init)(x) if self.perturbation in [3, 4] else Dense(units=self.hidden_layers_a[i], kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(l=self.l2_reg_a))(x)
            if int(self.BN_a[i]): x = BatchNormalization()(x)
            if self.perturbation == 2: x = LayerNormalization()(x)
            x = Activation(self.activ_a[0])(x)
        x = NoisyDense(units=self.dim_actions, sigma_init=self.sigma_init)(x) if self.perturbation in [3, 4] else Dense(units=self.dim_actions, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(l=self.l2_reg_a))(x)
        if int(self.BN_a[-1]): x = BatchNormalization()(x)
        if self.perturbation == 2: x = LayerNormalization()(x)
        output_layer = Activation(self.activ_a[-1])(x)

        model = Model(inputs = [input_layer], outputs = [output_layer])
        # print('ActorNetwork Summary:')
        # model.summary()
        return model
          

    def build_critic_network(self): # Critic
        input_state_layer = Input(shape=(self.dim_states,))
        input_action_layer = Input(shape=(self.dim_actions,))
        if self.TD3: 
            x = Concatenate(axis=-1)([input_state_layer, input_action_layer]) # axis=-1 -> last axis
            x = Dense(units=self.hidden_layers_c[0], kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(l=self.l2_reg_c))(x)
        else:
            x = Dense(units=self.hidden_layers_c[0], kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(l=self.l2_reg_c))(input_state_layer)
        if int(self.BN_c[0]): x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Concatenate(axis=-1)([x, input_action_layer])
        for i in range(1, len(self.hidden_layers_c)):
            x = Dense(units=self.hidden_layers_c[i], kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(l=self.l2_reg_c))(x)
            if int(self.BN_c[i]): x = BatchNormalization()(x)
            x = Activation("relu")(x)
        x = Dense(units=1, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(l=self.l2_reg_c))(x)
        if int(self.BN_c[-1]): x = BatchNormalization()(x)
        output_layer = Activation("linear")(x)

        model = Model(inputs = [input_state_layer, input_action_layer], outputs = [output_layer])
        # print('CriticNetwork Summary:')
        # model.summary()
        return model


    def sync_target_actor_network(self):
        weights = self.actor_network.get_weights()
        target_weights = self.target_actor_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - self.tau)
            target_weights[idx] += self.tau * w
        self.target_actor_network.set_weights(target_weights)


    def sync_target_critic_network(self):
        weights = self.critic_network.get_weights()
        target_weights = self.target_critic_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - self.tau)
            target_weights[idx] += self.tau * w
        self.target_critic_network.set_weights(target_weights)


    def sync_target_critic_2_network(self): #TD3
        weights = self.critic_2_network.get_weights()
        target_weights = self.target_critic_2_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - self.tau)
            target_weights[idx] += self.tau * w
        self.target_critic_2_network.set_weights(target_weights)


    # def build_forward(self): # from ICM
    #     input_state_layer = Input(shape=(self.dim_states,))
    #     input_action_layer = Input(shape=(self.dim_actions,))
    #     x = Concatenate(axis=-1)([input_state_layer, input_action_layer])
    #     x = Dense(units=self.hidden_layers_f[0], kernel_initializer=self.kernel_initializer)(x)
    #     x = Activation("relu")(x)
    #     for i in range(1, len(self.hidden_layers_f)):
    #         x = Dense(units=self.hidden_layers_f[i], kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(l=self.l2_reg_c))(x)
    #         # if int(self.BN_f[i]): x = BatchNormalization()(x)
    #         x = Activation("relu")(x)
    #     x = Dense(units=self.dim_states, kernel_initializer='glorot_normal')(x)
    #     # if int(self.BN_f[-1]): x = BatchNormalization()(x)
    #     output_layer = Activation("linear")(x)

    #     model = Model(inputs = [input_action_layer, input_state_layer], outputs = output_layer)
    #     # model = Model(inputs = input_layer, outputs = output_layer)
    #     #print('ICM Forward Summary:')
    #     #model.summary()
    #     return model


    def action_predict(self, state, epsilon, phase='learning'):
        state = state.reshape((1,-1))
        output = self.actor_network.predict(state)
        if phase == 'test':
            noise = np.zeros_like(output)
        elif self.perturbation == 0:
            noise = np.random.normal(loc=0.0, 
                                     scale=self.sigma, 
                                     size=output.shape)
        elif self.perturbation == 1:
            noise = self.OU(output)
        else:
            noise = np.zeros_like(output)
        self.action_space_noise = max(epsilon, 0) * noise
        action = output + self.action_space_noise
        action *= self.range_action_range
        action += self.range_action_mean
        action = np.clip(action, self.range_action_low, self.range_action_high)[0]
        return action


    def action_normalized(self, action):
        action -= self.range_action_mean
        action /= self.range_action_range
        return action


    def OU(self, x):
        ou_noise_prev = np.zeros_like(x) if self.ou_noise is None else self.ou_noise
        self.ou_noise = ou_noise_prev + \
                        self.theta * (self.mu - ou_noise_prev) * self.dt + \
                        self.sigma * np.random.normal(size=x.shape) * np.sqrt(self.dt)
        return self.ou_noise
        #-------------------------------
        # dx(t)=theta*(mu -x_(t))*dt+sigma*dW(t)
        # W(t) denotes the Wiener process.
        # W(t)=W(t)-W(0) =? N(0,t).
        # A corollary useful for simulation is that we can write, for t1 < t2:
        # W(t(2))=W(t(1))+sqrt(t(2)-t(1)) * Z
        # where Z is an independent standard normal variable.
        #-------------------------------


    def parameter_noise_update(self, states, target_sigma=None, phase='learning'):
        if phase == 'resume':
            for layer in self.actor_network.layers:
                if "noisy_dense" in layer.name:
                    prev_weights = self.prev_weights_dict[layer.name]
                    layer.set_weights(prev_weights)         
            return None   
        else:
            preturbated_actor_output = self.actor_network.predict(states)
            # Remove Noise
            for layer in self.actor_network.layers:
                if "noisy_dense" in layer.name:
                    new_weights = []
                    prev_weights = []
                    for i ,w in enumerate(layer.weights):
                        prev_weights.insert(i, layer.get_weights()[i])
                        if w in self.actor_network.non_trainable_weights:
                            new_weights.insert(i, np.zeros_like(layer.get_weights()[i]))
                        else:
                            new_weights.insert(i, layer.get_weights()[i])
                    self.prev_weights_dict[layer.name] = prev_weights
                    layer.set_weights(new_weights)
            non_preturbated_actor_output = self.actor_network.predict(states)
                                
            # Manually control the scale (if the target is set)
            distance = np.sqrt(np.power(preturbated_actor_output - non_preturbated_actor_output, 2).mean())      
            if target_sigma is None:
                pass
            elif target_sigma < distance:
                self.paramas_noise_sigma /= 1.01
            elif target_sigma > distance:
                self.paramas_noise_sigma *= 1.01
                
            # Update Noise  
            if phase == 'test':
                pass
            else:                        
                for layer in self.actor_network.layers:
                    if "noisy_dense" in layer.name:
                        new_weights = []
                        prev_weights = self.prev_weights_dict[layer.name]
                        for i ,w in enumerate(layer.weights):
                            if w in self.actor_network.non_trainable_weights:
                                if self.perturbation == 3:
                                    new_weights.insert(i, np.random.normal(loc=0.0, scale=self.paramas_noise_sigma, size=layer.get_weights()[i].shape))
                                elif self.perturbation == 4:
                                    ou_noise = prev_weights[i] + self.theta * (self.mu - prev_weights[i]) * self.dt + self.sigma * np.random.normal(size=layer.get_weights()[i].shape) * np.sqrt(self.dt)
                                    new_weights.insert(i, ou_noise)
                                else:
                                    ValueError
                            else:
                                new_weights.insert(i, layer.get_weights()[i])
                        layer.set_weights(new_weights)
            
            return distance


    def HiddenLayersSelector(self, choice):
        if   choice == 0: hidden_layers = [20, 15, 10]
        elif choice == 1: hidden_layers = [40, 30, 20]
        elif choice == 2: hidden_layers = [64, 64]
        elif choice == 3: hidden_layers = [100, 100]
        elif choice == 4: hidden_layers = [100, 50, 25]
        elif choice == 5: hidden_layers = [400, 300]
        elif choice == 6: hidden_layers = [400, 300, 300]
        elif choice == 7: hidden_layers = [400, 300, 300, 300]
        return hidden_layers


    def HiddenLayersCreator(self, choice):
        base_seeds = [4, 3, 2, 1, 0]
        mult = [5, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100]
        base_layers_dict = {}
        for base_layers in itertools.combinations_with_replacement(base_seeds, choice[2]-1):
            base_layers = [i for i in base_layers if i != 0]
            base_layers = np.array(base_layers)
            base_layers = np.insert(base_layers, 0, 4)  
            num_p = mult[0] * base_layers[0] + mult[0] * base_layers[0]
            for j in range(1, len(base_layers)):
                num_p += mult[0] * base_layers[j-1] * mult[0] * base_layers[j] + mult[0] * base_layers[j]
            base_layers_dict[num_p+base_layers.size/10] = base_layers
        base_layers_dict_keys = list(base_layers_dict.keys())
        base_layers_dict_keys.sort()
        base_layers_list = []
        for k in base_layers_dict_keys:
            base_layers_list.append(base_layers_dict[k])
        base_layers_list = base_layers_list[1:]
        hidden_layers = base_layers_list[choice[0]] * mult[choice[1]]
        return hidden_layers
