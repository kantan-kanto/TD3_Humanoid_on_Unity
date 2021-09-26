import sys, os, gym, pybullet_envs, optuna, traceback, logging, warnings, math, pathlib
import numpy as np
import random as rn
import sqlite3 as sql
import pandas as pd
from collections import namedtuple, deque
from datetime import datetime
from pandas import to_pickle, read_pickle
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from model import DDPG
from logger import TensorBoardLogger, Histograms
from SumTree import SumTree
from time import sleep
import socket, struct


model = 'DDPG'
env_id = 'Humanoid'
study_name = '00'
n_trials = 5
step = None # To pickle memory


def objective():
    #-------------------------------
    # Socket
    #-------------------------------   
    s = socket.socket()
    socket.setdefaulttimeout(None)
    port = 60000
    for i in range(1, 11):
        try:
            s.bind(('127.0.0.1', port))
        except Exception as e:
            print("error:{e} retry:{i}/{max}".format(e=e, i=i, max=10))
            sleep(i * 5)
            continue
        else:
            break
        sys.exit()
    s.listen()
    print('socket listensing ... ')

    #-------------------------------
    # Environment Setting
    #-------------------------------   
    run_id = 0 #trial.trial_id #trial.number #datetime.now().strftime('%Y%m%d%H%M%S-%f') 
    save = False
    R_note = 1000
    load = False


    dim_states = 32 #env.observation_space.shape[0]
    dim_actions = 11 #env.action_space.shape[0]
    range_action_high = np.ones(dim_actions) #env.action_space.high
    range_action_low = - np.ones(dim_actions) 
    range_action_range = (range_action_high - range_action_low) / 2.
    range_action_mean = (range_action_high + range_action_low) / 2.

    if env_id == 'Humanoid':
        log_interval = 1
        target_criteria = 'sma' #'max'
        target_episode_start = 0
        target_episode_end = 30000 #30000

        stage0_episode_start = 15000
        stage1_episode_start = 20000
        stage2_episode_start = 25000

        stage0_sma_interval = 10000
        stage1_sma_interval = 10000
        stage2_sma_interval = 10000

        stage0_reward_threshold = 200
        stage1_reward_threshold = 400
        stage2_reward_threshold = 1000

    #-------------------------------
    # Hyperparameter to be searched
    #-------------------------------
    max_memory_size = 1000000
    min_memory_size = 10000
    nb_test_episodes = 100
    perturbation_dict = {'gaussian': 0, 'ou': 1, 'layer_normalization': 2, 'noisy_dense' : 3, 'ou_noisy_dense' : 4}

    if len(study_name) == 2:
        perturbation = 0 #perturbation_dict['noisy_dense']
        PER = False
        TD3 = True
        fix_seed = True

        if TD3:
            hidden_layers_a = 5 #trial.suggest_int('hidden_layers_a', 1, 6) if l_lim else None
            hidden_layers_c = 5 #trial.suggest_int('hidden_layers_c', 5, 6) if l_lim else None
            lr_a = 10 ** -3 #trial.suggest_discrete_uniform('lr_a', -4.5, -3, 0.5)
            lr_c = 10 ** -3 #lr_a * 10 ** trial.suggest_discrete_uniform('lr_c_mult', 0, 0.5, 0.5)
            tau = 5 * 10 ** -3 #trial.suggest_int('tau',-4, -2)
            batch_size = 100 #2 ** trial.suggest_int('batch_size', 5, 7)
            sigma =  0.1 #trial.suggest_discrete_uniform('sigma', 0.1, 1.0, 0.05) if perturbation in [1, 4] else 0.1

        random_seed = 2 #trial.suggest_int('random_seed', 0, 9) if fix_seed else None
        rescale = 10 ** 0 #trial.suggest_discrete_uniform('rescale', -2, 0, 0.5)
        epsilon = 1 #trial.suggest_discrete_uniform('epsilon', 1.0, 1.0, 0.1) if perturbation in [0, 1] else 1
        EXPLORE = 10 ** 6 #trial.suggest_discrete_uniform('EXPLORE', 6, 8, 0.5) if perturbation in [0, 1] else 1
        theta = 0 #trial.suggest_discrete_uniform('theta', 0.1, 0.9, 0.05) if perturbation in [1, 4] else 
        sigma_init = 0.017 if perturbation == 3 else None
        nb_rollout_steps = 1 #int(2 ** trial.suggest_discrete_uniform('nb_rollout_steps', 8, 10, 1)) if perturbation in [3, 4] else 

    #-------------------------------
    # Hyperparameter future work
    #-------------------------------
    if True:
        prefill_buffer = True

        l2_reg_a = 0 # NOT NONE!
        l2_reg_c = 0 # NOT NONE!    

        if PER:
            PER_mode = 2 #trial.suggest_int('PER_mode', 1, 3)
            PER_weights = 1 if PER_mode >= 0 else 0
            PER_max = 1 if PER_mode >= 1 else 0
            PER_anneal = 1 if PER_mode >= 2 else 0
            max_memory_size = 10 ** 5 if PER_mode >= 3 else 10 ** 6

            per_a = trial.suggest_discrete_uniform('per_a', 0, 1, 0.2)
            per_b_init = trial.suggest_discrete_uniform('per_b', 0, 1, 0.2)
            per_b_grad = (1.0 - per_b_init) / target_episode_end if PER_anneal else 0
            per_e = 10 ** trial.suggest_discrete_uniform('per_e', -3, -1, 1)

        BN_a = 16
        BN_c = 16

        # hidden_layers_f = 0
        # beta = 0
        # lr_f = 10 ** -3

        if fix_seed:
            #-------------------------------
            # Fix random seed
            #-------------------------------
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            np.random.seed(random_seed)
            rn.seed(random_seed)
            # env.seed(random_seed)
            tf.reset_default_graph()
            config = tf.ConfigProto(
                intra_op_parallelism_threads=1, 
                inter_op_parallelism_threads=1,
                gpu_options=tf.GPUOptions(allow_growth=True),
                )
            tf.set_random_seed(random_seed)
            sess = tf.Session(
                graph=tf.get_default_graph(),
                config=config,
                )
            K.set_session(sess)
        else:
            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True),
                )
            sess = tf.Session(
                config=config,
                )
            K.set_session(sess)            

    #-------------------------------
    # Initialize & Load model
    #-------------------------------
    if len(study_name) == 2:
        log_dir = "./log/{}-{}-{}/{}".format(model, env_id, study_name, run_id)
    else:
        log_dir = "./log/{}-{}-{}/{}_{}".format(model, env_id, study_name[:2], study_name[3:], run_id)
    logger = TensorBoardLogger(log_dir=log_dir)
    histograms = {}

    num_episode = 1
    num_step = 0
    num_total_step = 0

    sum_reward = 0
    # sum_icm_reward = 0

    R_s0 = np.nan
    R_s1 = np.nan
    R_s2 = np.nan

    td_loss = 0
    nabla_J = 0
    # icm_loss = 0

    list_R_log = []
    # list_Ri_log = []
    list_R_all = []
    
    list_R_s0 = deque([], maxlen=stage0_sma_interval)
    list_R_s1 = deque([], maxlen=stage1_sma_interval)
    list_R_s2 = deque([], maxlen=stage2_sma_interval)

    # list_td_loss = deque([], maxlen=max_episode_steps*log_interval)
    # list_nabla_J = deque([], maxlen=max_episode_steps*log_interval)
    # list_Q = deque([], maxlen=max_episode_steps*log_interval)

    memory = SumTree(max_memory_size) if PER else deque([], maxlen=max_memory_size)
    num_memory = memory.len if PER else len(memory)
    
    if PER:
        # global Step
        Step = namedtuple("step", [
                                "state", 
                                "action", 
                                "next_reward", 
                                "next_state", 
                                # "icm_reward", 
                                "done",
                                ])    
    else:
        global step
        step = namedtuple("step", [
                                "state", 
                                "action", 
                                "next_reward", 
                                "next_state", 
                                # "icm_reward", 
                                "done",
                                ])

    agent = DDPG(
                dim_states=dim_states, 
                dim_actions=dim_actions,
                range_action_high=range_action_high,
                range_action_low=range_action_low,
                hidden_layers_a=hidden_layers_a,
                hidden_layers_c=hidden_layers_c,
                kernel_initializer='he_normal',
                l2_reg_a=l2_reg_a,
                l2_reg_c=l2_reg_c,
                BN_a=BN_a,
                BN_c=BN_c,
                activ_a=['relu', 'tanh'],
                optim_a='Adam',
                optim_c='Adam',
                lr_a=lr_a,
                lr_c=lr_c,
                tau=tau,
                perturbation=perturbation,
                theta=theta,
                sigma_init=sigma_init,
                sigma=sigma,
                # hidden_layers_f=hidden_layers_f,
                # lr_f=lr_f,
                TD3=TD3,
                # icm=icm,
                )

    if load:
        agent.actor_network_model = load_model(("./result/{}/{}/{}_actor_model.h5").format(env_id, study_name, model))
        agent.critic_network_model = load_model(("./result/{}/{}/{}_critic_model.h5").format(env_id, study_name, model))
        agent.actor_network = load_model(("./result/{}/{}/{}_actor_network.h5").format(env_id, study_name, model))
        agent.critic_network = load_model(("./result/{}/{}/{}_critic_network.h5").format(env_id, study_name, model))
        agent.target_actor_network = load_model(("./result/{}/{}/{}_target_actor_network.h5").format(env_id, study_name, model))
        agent.target_critic_network = load_model(("./result/{}/{}/{}_target_critic_network.h5").format(env_id, study_name, model))
        if TD3:
            agent.critic_2_network_model = load_model(("./result/{}/{}/{}_critic_2_model.h5").format(env_id, study_name, model))
            agent.critic_2_network = load_model(("./result/{}/{}/{}_critic_2_network.h5").format(env_id, study_name, model))
            agent.target_critic_2_network = load_model(("./result/{}/{}/{}_target_critic_2_network.h5").format(env_id, study_name, model))
        # if icm: agent.forward_model = load_model(("./result/{}/{}/{}_forward_model.h5").format(env_id, study_name, model))
        if not PER: memory = read_pickle(("./result/{}/{}/{}_memory.pkl").format(env_id, study_name, model))
        print('Pre-trained model loaded, run id:{} ready.'.format(run_id))
    else:
        print('Model loaded, run id:{} ready.'.format(run_id))

    #-------------------------------
    # Agent-Environment interaction cycle
    #-------------------------------
    start_time = datetime.now()
    try:
        #-------------------------------
        # Fill buffer up to min_memory_size
        #-------------------------------
        while (num_memory < min_memory_size and prefill_buffer and load == False):
                #-------------------------------
                # Receive states from Unity
                #------------------------------- 
                c, addr = s.accept()
                bytes_received = c.recv(4000)
                array_received = np.frombuffer(bytes_received, dtype=np.float32)

                if num_step:
                    #-------------------------------
                    # In the middle of a episode, 2/2
                    #-------------------------------
                    next_state = np.clip(array_received[:-2], -5, 5) # clip from -5  to 5 
                    next_reward = array_received[-2]
                    done = array_received[-1]
                    action = agent.action_normalized(action)
                    sum_reward += next_reward
                    next_reward = rescale * next_reward
                    # if icm:
                    #     next_state_pred = agent.forward_model.predict([state.reshape(1, -1), action.reshape(1, -1)])
                    #     icm_reward = beta * np.sum(np.square(next_state_pred - next_state)) / (2 * dim_states)
                    # else:
                    #     icm_reward = 0
                    # sum_icm_reward += icm_reward
                    
                    if PER:
                        action_pred = agent.actor_network.predict(next_state.reshape(1, -1))
                        next_target = agent.target_critic_network.predict([next_state.reshape(1, -1), action_pred])
                        if TD3:
                            next_target_2 = agent.target_critic_2_network.predict([next_state.reshape(1, -1), action_pred])
                            next_target = np.minimum(next_target, next_target_2)
                        next_target = next_reward + agent.gamma * next_target * (1 - done)
                        target = agent.critic_network.predict([state.reshape(1, -1), action.reshape(1, -1)])
                        td_error = np.abs(next_target - target) + per_e
                        priority = np.power(td_error, per_a)[0][0]
                        memory.add(
                            priority, 
                            Step(
                                state=state,
                                action=action,
                                next_reward=next_reward,
                                next_state=next_state,
                                # icm_reward=icm_reward,
                                done=done)
                            )
                    else:
                        memory.append((
                            None,
                            None,
                            step(
                                state=state,
                                action=action,
                                next_reward=next_reward,
                                next_state=next_state,
                                # icm_reward=icm_reward,
                                done=done)
                            ))
                    num_memory = memory.len if PER else len(memory)
                else:
                    #-------------------------------
                    # Start a episode
                    #------------------------------- 
                    # next_state, done = env.reset(), False
                    next_state = np.clip(array_received[:-2], -5, 5) # clip from -5  to 5 
                    # next_reward = array_received[-2]
                    done = array_received[-1]

                if done == 9: # achieve the goal
                    print("learning completed ")
                    c.close()
                    sys.exit()
                elif done == 1:
                    #-------------------------------
                    # End a step
                    #------------------------------- 
                    sum_reward = 0
                    # sum_icm_reward = 0
                    num_step = 0
                elif done == 0: # Episode on going
                    #-------------------------------
                    # In the middle of a episode, 1/2
                    #-------------------------------
                    num_step += 1
                    num_total_step += 1
                    state = next_state
                    if model == 'DDPG':
                        action = np.clip(np.random.normal(loc=0.0, scale=1.0, size=dim_actions), -1, 1)
                        # action = agent.action_predict(state, epsilon, phase='learning')
                        # if epsilon > 0 and perturbation in [0, 1]: epsilon -= 1.0 / EXPLORE
                    elif model == 'GP':
                        action = agent.action_predict(state)

                    if num_memory % 1000 == 0:
                        elapsed_time = (datetime.now() - start_time)
                        print(
                            "Filling memory buffer... {:>2d}/10  time:{}"
                            .format(num_memory * 10 // min_memory_size, str(elapsed_time)[:-7])
                            )

                #-------------------------------    
                # Send Action to Unity
                #------------------------------- 
                nn_output = np.array(action, dtype = 'float')
                #nn_output = np.array([action], dtype = 'float')
                bytes_to_send = struct.pack('%sf' % len(nn_output), *nn_output)
                c.sendall(bytes_to_send)
                c.close()

        #-------------------------------
        # Start learning
        #-------------------------------    
        while (
                math.isnan(nabla_J) == False and
                (num_episode <= stage0_episode_start or math.isnan(R_s0) or R_s0 > stage0_reward_threshold) and
                (num_episode <= stage1_episode_start or math.isnan(R_s1) or R_s1 > stage1_reward_threshold) and
                (num_episode <= stage2_episode_start or math.isnan(R_s2) or R_s2 > stage2_reward_threshold) and
                num_episode <= target_episode_end
                ):
            #-------------------------------
            # Receive states from Unity
            #-------------------------------             
            c, addr = s.accept()
            bytes_received = c.recv(4000)
            array_received = np.frombuffer(bytes_received, dtype=np.float32)

            if num_step:
                #-------------------------------
                # In the middle of a episode, 2/2
                #-------------------------------
                next_state = np.clip(array_received[:-2], -5, 5) # clip from -5  to 5 
                next_reward = array_received[-2]
                done = array_received[-1]
                action = agent.action_normalized(action)
                sum_reward += next_reward
                next_reward = rescale * next_reward
                # if icm:
                #     next_state_pred = agent.forward_model.predict([state.reshape(1, -1), action.reshape(1, -1)])
                #     icm_reward = beta * np.sum(np.square(next_state_pred - next_state)) / (2 * dim_states)
                # else:
                #     icm_reward = 0
                # sum_icm_reward += icm_reward
                
                if PER:
                    action_pred = agent.actor_network.predict(next_state.reshape(1, -1))                   
                    next_target = agent.target_critic_network.predict([next_state.reshape(1, -1), action_pred])
                    if TD3:
                        next_target_2 = agent.target_critic_2_network.predict([next_state.reshape(1, -1), action_pred])
                        next_target = np.minimum(next_target, next_target_2)
                    next_target = next_reward + agent.gamma * next_target * (1 - done)
                    target = agent.critic_network.predict([state.reshape(1, -1), action.reshape(1, -1)])
                    td_error = np.abs(next_target - target) + per_e
                    priority = np.power(td_error, per_a)[0][0]
                    memory.add(
                        priority, 
                        Step(
                            state=state,
                            action=action,
                            next_reward=next_reward,
                            next_state=next_state,
                            # icm_reward=icm_reward,
                            done=done)
                        )
                else:
                    memory.append((
                        None,
                        None,
                        step(
                            state=state,
                            action=action,
                            next_reward=next_reward,
                            next_state=next_state,
                            # icm_reward=icm_reward,
                            done=done)
                        ))
                num_memory = memory.len if PER else len(memory)
            else:
                #-------------------------------
                # Start a episode
                #-------------------------------
                # next_state, done = env.reset(), False
                next_state = np.clip(array_received[:-2], -5, 5) # clip from -5  to 5 
                # next_reward = array_received[-2]
                done = array_received[-1]

            #-------------------------------
            # Train
            #------------------------------- 
            if PER:
                batch = []
                segment = memory.total() / batch_size
                for i in range(batch_size):
                    a = segment * i
                    b = segment * (i + 1)
                    s = rn.uniform(a, b)
                    (idx, priority, step) = memory.get(s)
                    batch.append((idx, priority, step))
            else:
                batch = rn.sample(memory, batch_size)

            priorities = np.asarray([priority for _, priority, _ in batch]).reshape(-1, 1)
            states = np.asarray([step.state for _, _, step in batch]).reshape(-1, dim_states)
            actions = np.asarray([step.action for _, _, step in batch]).reshape(-1, dim_actions)
            next_rewards = np.asarray([step.next_reward for _, _, step in batch]).reshape(-1, 1)
            next_states = np.asarray([step.next_state for _, _, step in batch]).reshape(-1, dim_states)
            # icm_rewards = np.asarray([step.icm_reward for _, _, step in batch]).reshape(-1, 1)
            dones = np.asarray([step.done for _, _, step in batch]).reshape(-1, 1)                

            if model == 'DDPG':
                action_target_preds = agent.target_actor_network.predict(next_states)
                # Target Policy Smoothing Regularization
                if TD3: 
                    noise = np.random.normal(loc=0.0, 
                                             scale=0.2, 
                                             size=(batch_size, dim_actions))
                    noise = np.clip(noise, -0.5, 0.5)
                    action_target_preds += noise         
                next_targets = agent.target_critic_network.predict([next_states, action_target_preds])
                if TD3:
                    next_targets_2 = agent.target_critic_2_network.predict([next_states, action_target_preds])
                    next_targets = np.minimum(next_targets, next_targets_2)
            # elif model == 'GP':
            #     next_targets = agent.critic_network_model.predict(next_states)
            next_targets = next_rewards + agent.gamma * next_targets * (1 - dones)

            if PER:
                per_b = per_b_init + num_episode * per_b_grad
                probabilities = priorities / memory.total()
                weights = 1 / np.power(num_memory * probabilities, per_b) if PER_weights else np.ones_like(next_targets)
                max_weight = np.max(weights) if PER_max else 1
            weights = weights / max_weight if PER else np.ones_like(next_targets)
            # dummy = np.zeros_like(next_targets)
            if model == 'DDPG':
                next_targets_merged = np.concatenate([next_targets, weights], axis=1).reshape(batch_size, 1, -1)
                states_reshape = states.reshape(batch_size, 1, -1)
                td_loss = agent.critic_network_model.train_on_batch([states, actions], next_targets_merged)
                if TD3:
                    td_loss_2 = agent.critic_2_network_model.train_on_batch([states, actions], next_targets_merged)
                    # td_loss = min(td_loss, td_loss_2)
                    agent.sync_target_critic_2_network()
                agent.sync_target_critic_network()
                if num_total_step % 2 == 0 or TD3 == False:
                    nabla_J = -agent.actor_network_model.train_on_batch(states, states_reshape)
                    agent.sync_target_actor_network()
                # td_loss = agent.critic_network_model.train_on_batch([states, actions, weights], next_targets)
                # nabla_J = -agent.actor_network_model.train_on_batch(states, dummy)
            #     if icm: icm_loss = agent.forward_model.train_on_batch([states, actions], next_states)
                
            # elif model == 'GP':
            #     dummy = np.zeros_like(next_targets)
            #     td_loss = agent.critic_network_model.train_on_batch(states, next_targets)
            #     nabla_J = -agent.actor_network_model.train_on_batch([states, actions, next_targets], dummy)
            #     if icm: icm_loss = agent.forward_model.train_on_batch([states, actions], next_states)

            if PER:
                action_preds = agent.actor_network.predict(next_states)
                next_targets = agent.target_critic_network.predict([next_states, action_preds])
                if TD3:
                    next_targets_2 = agent.target_critic_2_network.predict([next_states, action_preds])
                    next_targets = np.minimum(next_targets, next_targets_2)
                next_targets = next_rewards + agent.gamma * next_targets * (1 - dones)
                targets = agent.critic_network.predict([states, actions])
                td_errors = np.abs(next_targets - targets) + per_e
                for i in range(len(batch)):
                    idx = batch[i][0]
                    priority = np.power(td_errors[i], per_a)[0]
                    memory.update(idx, priority)

            #-------------------------------
            # Update Parametere Noise
            #------------------------------- 
            if num_total_step % nb_rollout_steps == 0 \
                and perturbation in [3, 4] \
                and num_episode % nb_test_episodes != 0:

                distance = agent.parameter_noise_update(states, target_sigma=None, phase='learning') if perturbation in [3, 4] else 0
                logger.log(logs={"3 Noise/2 Distance": distance,
                                 "3 Noise/9 epsilon": epsilon},
                           epoch=num_total_step)

            if done == 9: # achieve the goal
                print("learning completed ")
                c.close()
                sys.exit()
            elif done == 1:
                #-------------------------------
                # End a episode
                #------------------------------- 
                list_R_log.append(sum_reward)
                # list_Ri_log.append(sum_icm_reward)
                list_R_all.append(sum_reward)
                
                if num_episode % nb_test_episodes == 0:
                    if perturbation in [3, 4]:
                        # for resume learning
                        agent.parameter_noise_update(states, target_sigma=None, phase='resume')
                else:
                    list_R_s0.append(sum_reward)
                    list_R_s1.append(sum_reward)
                    list_R_s2.append(sum_reward)
                
                #-------------------------------
                # log
                #-------------------------------
                elapsed_time = (datetime.now() - start_time)
                R_avg = sum(list_R_log)/len(list_R_log)
                # Ri_avg = sum(list_Ri_log)/len(list_Ri_log)

                if num_episode % nb_test_episodes == 0:
                    logger.log(logs={"2 Reward/2 Reward in test": R_avg},
                               epoch=num_episode)
                    print(
                        "Ep:{:>4d}  R:{:>4.0f}  (test phase)   time:{}"
                        .format(num_episode, R_avg, str(elapsed_time)[:-7])
                        )
                elif num_episode % log_interval == 0:
                  
                    R_s0 = max(list_R_s0) #sum(list_R_s0)/len(list_R_s0)
                    R_s1 = max(list_R_s1) #sum(list_R_s1)/len(list_R_s1)
                    R_s2 = max(list_R_s2) 

                    output_lr = {
                        "1 Loss/1 TD loss": td_loss,
                        "1 Loss/2 nabla J": nabla_J,
                        "2 Reward/1 Reward in training": R_avg,
                        # "3 Noise/9 ICM Reward avg": Ri_avg,
                        # "3 Noise/9 ICM loss": icm_loss,
                        "9 misc./1 Memory size": num_memory,
                        "9 misc./2 Total Step": num_total_step,
                    }
                    logger.log(logs=output_lr, epoch=num_episode)

                    print(
                        "Ep:{:>4d}  R:{:>4.0f}  nabla J:{:>4.3f}  time:{}"
                        .format(num_episode, R_avg, nabla_J, str(elapsed_time)[:-7])
                        )

                    # if len(study_name) == 2:
                    #     try:
                    #         study_best_value = study.best_value
                    #     except ValueError:
                    #         study_best_value = np.nan
                    # else:
                    #     study_best_value = np.nan
                    # if R_note < R_avg and (study_best_value < R_avg or math.isnan(study_best_value)): 
                    if (R_note < R_avg) and (save): 
                        R_note = R_avg
                        os.makedirs(("./result/{}/{}_{}").format(env_id, study_name, run_id), exist_ok=True)
                        agent.actor_network_model.save(("./result/{}/{}_{}/{}_actor_model.h5").format(env_id, study_name, run_id, model))
                        agent.critic_network_model.save(("./result/{}/{}_{}/{}_critic_model.h5").format(env_id, study_name, run_id, model))
                        agent.actor_network.save(("./result/{}/{}_{}/{}_actor_network.h5").format(env_id, study_name, run_id, model))
                        agent.critic_network.save(("./result/{}/{}_{}/{}_critic_network.h5").format(env_id, study_name, run_id, model))
                        agent.target_actor_network.save(("./result/{}/{}_{}/{}_target_actor_network.h5").format(env_id, study_name, run_id, model))
                        agent.target_critic_network.save(("./result/{}/{}_{}/{}_target_critic_network.h5").format(env_id, study_name, run_id, model))
                        if TD3:
                            agent.critic_2_network_model.save(("./result/{}/{}_{}/{}_critic_2_model.h5").format(env_id, study_name, run_id, model))
                            agent.critic_2_network.save(("./result/{}/{}_{}/{}_critic_2_network.h5").format(env_id, study_name, run_id, model))
                            agent.target_critic_2_network.save(("./result/{}/{}_{}/{}_target_critic_2_network.h5").format(env_id, study_name, run_id, model))
                        # if icm: agent.forward_model.save(("./result/{}/{}_{}/{}_forward_model.h5").format(env_id, study_name, run_id, model))
                        if not PER: to_pickle(memory, ("./result/{}/{}_{}/{}_memory.pkl").format(env_id, study_name, run_id, model))
                        pathlib.Path(("./result/{}/{}_{}/R_{}-ep_{}").format(env_id, study_name, run_id, int(R_avg), num_episode)).touch()
             
                if num_episode % 1000 == 0:
                    histograms = Histograms(model=agent.actor_network.layers, model_id="actor", histograms=histograms)
                    histograms = Histograms(model=agent.critic_network.layers, model_id="critic", histograms=histograms)
                    # if icm:
                    #     histograms = Histograms(model=agent.forward.layers, model_id="forward", histograms=histograms)
                    logger.log(histograms=histograms, epoch=num_episode)
                    histograms = {}

                sum_reward = 0
                # sum_icm_reward = 0
                num_step = 0
                num_episode += 1                 
                list_R_log = []
                # list_Ri_log = []

                if num_episode % nb_test_episodes == 0 and perturbation in [3, 4]:
                    # for starting test phase
                    agent.parameter_noise_update(states, target_sigma=None, phase='test')
            elif done == 0: # Episode on going
                #-------------------------------
                # In the middle of a episode, 1/2
                #-------------------------------
                num_step += 1
                if num_episode % nb_test_episodes != 0: num_total_step += 1
                state = next_state
                if model == 'DDPG':
                    if num_episode % nb_test_episodes == 0:
                        action = agent.action_predict(state, epsilon, phase='test')
                    else:
                        action = agent.action_predict(state, epsilon, phase='learning')
                        if epsilon > 0 and perturbation in [0, 1]:
                            epsilon -= 1.0 / EXPLORE
                            if num_total_step % 100 == 0 and perturbation in [0, 1]:
                                logger.log(logs={"3 Noise/1 Action Space Noise": agent.action_space_noise[0][0],
                                                 "3 Noise/9 epsilon": epsilon}, 
                                           epoch=num_total_step)
                # elif model == 'GP':
                #     action = agent.action_predict(state)
                # if action_max < action: action_max = action
                # if action_min > action: action_min = action      
                
            #-------------------------------    
            # Send Action to Unity
            #------------------------------- 
            nn_output = np.array(action, dtype = 'float')
            #nn_output = np.array([action], dtype = 'float')
            bytes_to_send = struct.pack('%sf' % len(nn_output), *nn_output)
            c.sendall(bytes_to_send)
            c.close()


    except KeyboardInterrupt:
        print('KeyboardInterrupt')


    except:
        traceback.print_exc()


    finally:
        #-------------------------------
        # Wrap up
        #-------------------------------
        # env.close()
        print("R_s0: {:>4.3f}   R_s1: {:>4.3f}   R_s2: {:>4.3f}".format(R_s0, R_s1, R_s2))
        
        target_sma_interval = {'max': 1, 'sma': 100, 'avg': target_episode_end - target_episode_start}.setdefault(target_criteria, 1)
        target_episode_start = min(target_episode_start, len(list_R_all) - 1)
        interval = np.ones(min(target_sma_interval, len(list_R_all[target_episode_start:])))
        interval /= len(interval)
        R_target = np.max(np.convolve(list_R_all[target_episode_start:], interval, mode='valid'))
        return R_target


if __name__ == '__main__':
    #-------------------------------
    # Ignore Warning
    #-------------------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    tf.get_logger().setLevel('INFO')
    # tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    # K.set_learning_phase(0) # 0 = test, 1 = train

    objective()
