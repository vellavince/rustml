use rustml::*;
use tch::{Tensor, Kind};

struct Policy {
    l1: Linear,
    l2: Linear,
}

impl Policy {
    fn new (mem: &mut Memory, nfeatures: i64, nactions: i64) -> Policy {
        let l1 = Linear::new(mem, nfeatures, 128);
        let l2 = Linear::new(mem, 128, nactions);

        Self {
            l1: l1,
            l2: l2,
        }
    }
}

impl Compute for Policy {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(mem, input);
        o = o.tanh();
        o = self.l2.forward(mem, &o);
        o
    }
}

fn main() {
    const MEM_SIZE: usize = 30000;
    const MIN_MEM_SIZE: usize = 5000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: i64 = 50;
    const LEARNING_RATE: f32 = 0.00005;
    let mut epsilon:f32 = 1.0;

    let mut state: Tensor;
    let mut action: i64;
    let mut reward:f32;
    let mut done: bool;
    let mut state_: Tensor;

    let mut mem_replay = ReplayMemory::new(MEM_SIZE, MIN_MEM_SIZE);
    let mut mem_policy = Memory::new();
    let policy_net = Policy::new(&mut mem_policy, 8, 4);
    let mut mem_target = Memory::new();
    let target_net = Policy::new(&mut mem_target, 8, 4);
    mem_target.copy(&mem_policy);
    let mut ep_returns = RunningStat::<f32>::new(50);
    let mut ep_return: f32 = 0.0;
    let mut nepisodes = 0;
    let one:Tensor = Tensor::from(1.0);

    gym_make("LunarLander-v2", false);
    state = reset();
    mem_replay.init();
    loop {
        action = epsilon_greedy(&mem_policy, &policy_net, epsilon, &state);
        (state_, reward, done) = step(action);
        let t = Transition::new(&state, action, reward, done, &state_);
        mem_replay.add(t);
        ep_return += reward;
        state = state_;

        if done {
            nepisodes += 1;
            ep_returns.add(ep_return);
            ep_return = 0.0;
            state = reset();

            let avg = ep_returns.average(); // sum() / 50.0;
            if nepisodes % 100 == 0 {
                println!("Episode: {}, Avg Return: {} Epsilon: {}", nepisodes, avg, epsilon);
            }
            if avg >= 180.0 {
                println!("Solved at episode {}", nepisodes);
                break;
            }
            epsilon = epsilon_update(avg, -300.0, 150.0, 0.05, 1.0);
        }

        let (b_state, b_action, b_reward, b_done, b_state_) = mem_replay.sample_batch(128);
        let qvalues = policy_net.forward(&mem_policy, &b_state).gather(1, &b_action, false);

        let target_values: Tensor = tch::no_grad(|| target_net.forward(&mem_target, &b_state_));
        let max_target_values = target_values.max_dim(1, true).0; 
        let expected_values = b_reward + GAMMA * (&one - &b_done) * (&max_target_values);
        
        let loss = mse(&qvalues, &expected_values);
        loss.backward();
        mem_policy.apply_grads_adam(LEARNING_RATE);
        if nepisodes % UPDATE_FREQ == 0 {
            mem_target.copy(&mem_policy);
        }
    }

    // Test the trained agent and save video
    gym_make("LunarLander-v2", true);
    state = reset();
    loop {
        action = epsilon_greedy(&mem_policy, &policy_net, 0.0, &state);
        (state_, reward, done) = step(action);
        state = state_;
        if done {
            break;
        }
    } 
    play();

}
