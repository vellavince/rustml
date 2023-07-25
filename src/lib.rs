use tch::{Tensor, Kind, Device, vision};
use rand::Rng;
use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};
use pyo3::prelude::*;
use std::{collections::HashMap, collections::VecDeque};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// General Structures
/// 

pub trait Compute {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Tensor Memory

pub struct Memory {
    size: usize,
    pub values: Vec<Tensor>,
}

impl Memory {

    pub fn new() -> Self {
        let v = Vec::new();
        Self {size: 0,
            values: v}
    }

    pub fn copy(&mut self, sourcemem: &Memory) {
        let cp = sourcemem.values.iter().map(|t| {
            let c = t.copy().set_requires_grad(false);
            c
        }).collect();
        self.values = cp;
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn params_info(&self) {
        self.values
        .iter()
        .for_each(|t| 
            {
                if t.requires_grad() { 
                    println!("{:?}", t.size());
                } 
            
            });
    }

    fn push (&mut self, value: Tensor) -> usize {
        self.values.push(value);
        self.size += 1;
        self.size-1
    }

    fn new_push (&mut self, size: &[i64], requires_grad: bool) -> usize {
        let t = Tensor::randn(size, (Kind::Float, Device::Cpu)).requires_grad_(requires_grad);
        self.push(t)
    }

    pub fn get (&self, addr: &usize) -> &Tensor {
        &self.values[*addr]
    }

    pub fn set (&mut self, addr: &usize, value: Tensor) {
        self.values[*addr] = value;
    }

    pub fn apply_grads_sgd(&mut self, learning_rate: f32) {
        let mut g = Tensor::new();      
        self.values
        .iter_mut()
        .for_each(|t| {
            if t.requires_grad() {
                g = t.grad();
                t.set_data(&(t.data() - learning_rate*&g));
                t.zero_grad();
            }
        });
    }

    pub fn apply_grads_sgd_momentum(&mut self, learning_rate: f32) {
        let mut g: Tensor = Tensor::new();
        let mut velocity: Vec<Tensor>= zeros(&[self.size as i64]).split(1, 0);
        let mut vcounter = 0;
        const BETA:f32 = 0.9;
        
        self.values
        .iter_mut()
        .for_each(|t| {
            if t.requires_grad() {
                g = t.grad();
                velocity[vcounter] =  (BETA * &velocity[vcounter]) + (1.0 - BETA) * &g;
                t.set_data(&(t.data() - learning_rate * &velocity[vcounter]));
                t.zero_grad();         
            }
            vcounter += 1;
        });
    }

    pub fn apply_grads_adam(&mut self, learning_rate: f32) {
        let mut g = Tensor::new();
        const BETA:f32 = 0.9;

        let mut velocity = zeros(&[self.size as i64]).split(1, 0);
        let mut mom = zeros(&[self.size as i64]).split(1, 0);
        let mut vel_corr = zeros(&[self.size as i64]).split(1, 0);
        let mut mom_corr = zeros(&[self.size as i64]).split(1, 0);
        let mut counter = 0;

        self.values
        .iter_mut()
        .for_each(|t| {
            if t.requires_grad() {
                g = t.grad();
                g = g.clamp(-1, 1);
                mom[counter] = BETA * &mom[counter] + (1.0 - BETA) * &g;
                velocity[counter] = BETA * &velocity[counter] + (1.0 - BETA) * (&g.pow(&Tensor::from(2)));    
                mom_corr[counter] = &mom[counter]  / (Tensor::from(1.0 - BETA).pow(&Tensor::from(2)));
                vel_corr[counter] = &velocity[counter] / (Tensor::from(1.0 - BETA).pow(&Tensor::from(2)));

                t.set_data(&(t.data() - learning_rate * (&mom_corr[counter] / (&velocity[counter].sqrt() + 0.0000001))));
                t.zero_grad();
            }
            counter += 1;
        });

    }


}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// Linear Model

pub struct Linear {
    params: HashMap<String, usize>,
}

impl Linear {
    pub fn new (mem: &mut Memory, ninputs: i64, noutputs: i64) -> Self {
        let mut p = HashMap::new();
        p.insert("W".to_string(), mem.new_push(&[ninputs,noutputs], true));
        p.insert("b".to_string(), mem.new_push(&[1, noutputs], true));
        Self {
            params: p,
        }
    } 
}

impl Compute for Linear {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let w = mem.get(self.params.get(&"W".to_string()).unwrap());
        let b = mem.get(self.params.get(&"b".to_string()).unwrap());
        input.matmul(w) + b
    }
}

pub struct Conv2d {
    params: HashMap<String, usize>,
    stride: i64,
}

impl Conv2d {
    pub fn new (mem: &mut Memory, kernel_size: i64, in_channel: i64, out_channel: i64, stride: i64) -> Self {
        let mut p = HashMap::new();
        p.insert("kernel".to_string(), mem.new_push(&[out_channel, in_channel, kernel_size, kernel_size], true));
        p.insert("bias".to_string(), mem.push(Tensor::full(&[out_channel], 0.0, (Kind::Float, Device::Cpu)).requires_grad_(true)));
        Self {
            params: p,
            stride: stride,
        }
    } 
}

impl Compute for Conv2d {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let kernel = mem.get(self.params.get(&"kernel".to_string()).unwrap());
        let bias = mem.get(self.params.get(&"bias".to_string()).unwrap());
        input.conv2d(&kernel, Some(bias), &[self.stride], 0, &[1], 1)
    }
}

pub struct RNN {
    params: HashMap<String, usize>,
    out_seq_len: i64,
    linear_layer: bool,
    linear_out_size: i64,
}

impl RNN {
    pub fn new (mem: &mut Memory, input_size: i64, hidden_size: i64, linear_layer: bool, linear_out_size: i64, out_seq_len: i64) -> Self {
        let mut p = HashMap::new();
        p.insert("Wxh".to_string(), mem.new_push(&[input_size, hidden_size], true));
        p.insert("Whh".to_string(), mem.new_push(&[hidden_size, hidden_size], true));
        p.insert("bh".to_string(), mem.new_push(&[hidden_size], true));
        if linear_layer {
            p.insert("W".to_string(), mem.new_push(&[hidden_size, linear_out_size], true));
            p.insert("b".to_string(), mem.new_push(&[1, linear_out_size], true));
        }
        Self {
            params: p,
            out_seq_len: out_seq_len,
            linear_layer: linear_layer,
            linear_out_size: linear_out_size,
        }
    }

    pub fn set_h0 (&self, mem: &mut Memory, h0: Tensor) {
        let h0_addr = self.params["h0"];
        mem.set(&h0_addr, h0);
    } 
}

impl Compute for RNN {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let wxh = mem.get(self.params.get(&"Wxh".to_string()).unwrap());
        let whh = mem.get(self.params.get(&"Whh".to_string()).unwrap());
        let bh = mem.get(self.params.get(&"bh".to_string()).unwrap());
        let mut w = &Tensor::from(0.0);
        let mut b = &Tensor::from(0.0);
        if self.linear_layer {
            w = mem.get(self.params.get(&"W".to_string()).unwrap());
            b = mem.get(self.params.get(&"b".to_string()).unwrap());
        }
        let batchsize = input.size()[0]; // input = datapoints x timesteps x features
        let timesteps = input.size()[1];  
        let out_start = timesteps - self.out_seq_len;
                
        let mut h : Tensor;
        if let Some(h0_addr) = self.params.get(&"h0".to_string()) {
            h = mem.get(h0_addr).copy();
        } else {    
            h = zeros(&[batchsize, bh.size()[0]]);
        };
        let mut out: Vec<Tensor> = Vec::new();
        let mut out_h: Vec<Tensor> = Vec::new();
        for i in 0..timesteps {
            let row = input.narrow(1, i, 1).squeeze_dim(1);
            h = (row.matmul(wxh) + h.matmul(whh) + bh).tanh();
            out_h.push(h.copy());
            if self.linear_layer {
                out.push(h.matmul(w) + b);
            }
        }
        let output: &Vec<Tensor>;
        if self.linear_layer {
            output = &out;
        } else {
            output = &out_h;
        }
        let res = Tensor::concat(output.as_slice(), 1).reshape(&[batchsize, timesteps, -1]);
        res.narrow(1, out_start, timesteps-out_start)
        
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// General Functions


pub fn mse(target: &Tensor, pred: &Tensor) -> Tensor {
    pred.smooth_l1_loss(&target, tch::Reduction::Mean, 0.0)
    //(target - pred).square().mean(Kind::Float)
}

pub fn cross_entropy (target: &Tensor, pred: &Tensor) -> Tensor {
    let loss = pred.log_softmax(-1, Kind::Float).nll_loss(target);
    loss
}

fn get_batches(x: &Tensor, y: &Tensor, batch_size: i64, shuffle: bool) -> impl Iterator<Item = (Tensor, Tensor)> {
    let num_rows = x.size()[0];
    let num_batches = (num_rows + batch_size - 1) / batch_size;
    
    let indices = if shuffle {
        Tensor::randperm(num_rows as i64, (Kind::Int64, Device::Cpu))
    } else 
    {
        let rng = (0..num_rows).collect::<Vec<i64>>();
        Tensor::from_slice(&rng)
    };
    let x = x.index_select(0, &indices);
    let y = y.index_select(0, &indices);
    
    (0..num_batches).map(move |i| {
        let start = i * batch_size;
        let end = (start + batch_size).min(num_rows);
        let batchx: Tensor = x.narrow(0, start, end - start);
        let batchy: Tensor = y.narrow(0, start, end - start);
        (batchx, batchy)
    })
}

pub fn train<F>(mem: &mut Memory, x: &Tensor, y: &Tensor, model: &dyn Compute, epochs: i64, batch_size: i64, errfunc: F, learning_rate: f32) 
    where F: Fn(&Tensor, &Tensor)-> Tensor    
        {
        let mut error = Tensor::from(0.0);
        let mut batch_error = Tensor::from(0.0);
        for epoch in 0..epochs {
            batch_error = Tensor::from(0.0);
            for (batchx, batchy) in get_batches(&x, &y, batch_size, true) {
                let pred = model.forward(mem, &batchx);
                error = errfunc(&batchy, &pred);
                batch_error += error.detach();
                error.backward();
                mem.apply_grads_adam(learning_rate);              
            }

            println!("Epoch: {:?} Error: {:?}", epoch, batch_error/batch_size);
            
        }
}

pub fn load_mnist() -> (Tensor, Tensor) {
    let m = vision::mnist::load_dir("data").unwrap();
    let x = m.train_images;
    let y = m.train_labels;
    (x, y)
}

pub fn accuracy(target: &Tensor, pred: &Tensor) -> f64 {
    let yhat = pred.argmax(1,true).squeeze();
    let eq = target.eq_tensor(&yhat);
    let accuracy: f64 = (eq.sum(Kind::Int64) / target.size()[0]).double_value(&[]).into();
    accuracy
}

pub fn zeros(size: &[i64]) -> Tensor {
    Tensor::zeros(size, (Kind::Float, Device::Cpu))
}

pub fn rgb_to_grayscale(tensor: &Tensor) -> Tensor {
    let red_channel = tensor.get(0);
    let green_channel = tensor.get(1);
    let blue_channel = tensor.get(2);
    
    // Calculate the grayscale tensor using the luminance formula
    let grayscale = (red_channel * 0.2989) + (green_channel * 0.5870) + (blue_channel * 0.1140);
    
    // Return the grayscale tensor
    grayscale.unsqueeze(0)
}

pub fn generate_random_timeseries(num_points: usize, noise: f64) -> (Tensor, Tensor) {
    let mut rng = rand::thread_rng();
    let mut x_values = Vec::with_capacity(num_points);
    let mut y_values = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let noise_value = noise * rng.gen::<f64>();
        let x = i as f64 * 0.01;
        let y = x.sin() + noise_value;
        x_values.push(x);
        y_values.push(y);
    }
    let x = Tensor::from_slice(&x_values).to_kind(Kind::Float);
    let y = Tensor::from_slice(&y_values).to_kind(Kind::Float);

    (x, y)
}

pub fn data_prep_rnn (y: &Tensor, lag: i64) -> (Tensor, Tensor) {
    let data = convert_to_lagged_series(&y, lag).reshape(&[-1, lag, 1]);
    let target = data.select(1, lag-1);
    let input = data.slice(1, 0, lag-1, 1);

    (input, target)
}

pub fn convert_to_lagged_series(x: &Tensor, n: i64) -> Tensor {
    let num_samples = x.size()[0] - n + 1;
    x.unfold(0, n, 1)
        .view((num_samples, n))
        .copy()
}


pub fn plot_graph(x: &Tensor, y: &Tensor) {
    // Convert tensors to vectors
    let x_vec: Vec<f64> = x.iter::<f64>().unwrap().collect();
    let y_vec: Vec<f64> = y.iter::<f64>().unwrap().collect();

    let mut figure = Figure::new();

    figure.axes2d().points(&x_vec, &y_vec, &[Caption("Data"), Color("red")]);
    figure.set_title("Plot Title");

    figure.show().unwrap();
}

pub fn graph_compare(x: &Tensor, y1: &Tensor, y2: &Tensor) {
    // Convert tensors to vectors
    let x_vec: Vec<f64> = x.iter::<f64>().unwrap().collect();
    let y1_vec: Vec<f64> = y1.iter::<f64>().unwrap().collect();
    let y2_vec: Vec<f64> = y2.iter::<f64>().unwrap().collect();

    let mut figure = Figure::new();

    figure.axes2d()
    .set_y_range(Fix(-1.5), Fix(1.5))
    .points(&x_vec, &y1_vec, &[Color("red")]);
    figure.axes2d()
    .set_y_range(Fix(-1.5), Fix(1.5))
    .points(&x_vec, &y2_vec, &[Color("blue")]);

    figure.set_title("Comparison between Actual (red) and Predicted (blue)");
    figure.show().unwrap();
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// 
/// RL Functions
pub fn py_proc (cmd: &str) {
    Python::with_gil(|py| {
        py.run(cmd, None, None).unwrap();
    })
}

pub fn py_getf32(var: &str) -> f32
{
    Python::with_gil(|py: Python<'_>| {
        let result = py.eval(var, None, None).unwrap();
        let out: f32 = result.extract().unwrap();
        out
    })
}

pub fn py_getf32vec(var: &str) -> Tensor
{
    Python::with_gil(|py: Python<'_>| {
        let result = py.eval(var, None, None).unwrap();
        let out: Vec<f32> = result.extract().unwrap();
        let t = Tensor::from_slice(&out);
        t
    })
}

pub fn gym_make (game: &str, store_video: bool) {
    pyo3::prepare_freethreaded_python();
    py_proc("import warnings");
    py_proc("warnings.filterwarnings('ignore')");
    py_proc("import gymnasium as gym");    
    py_proc("import numpy as np");
    let cmd = format!("env = gym.make('{}', render_mode = 'rgb_array')", game);
    py_proc(cmd.as_str());
    if store_video {
        py_proc("import renderlab as rl");
        py_proc("env = rl.RenderFrame(env, './video')");
    }
}

pub fn play () {
    py_proc("env.play()");
}

pub fn reset() -> Tensor {
    py_proc("observation, info = env.reset(seed=42)");
    py_getf32vec("observation")
}

pub fn sample_discrete_action() -> i64 {
    py_proc("action = env.action_space.sample()");
    py_getf32("action") as i64
}

pub fn step (action: i64) -> (Tensor, f32, bool) {
    let cmd = format!("observation, reward, termination, truncation, _ = env.step({})", action);
    py_proc(&cmd.to_string());
    let state = py_getf32vec("observation");
    let reward = py_getf32("reward");    
    let termimantion = py_getf32("termination") != 0.0;
    let truncation = py_getf32("truncation") != 0.0;
    let done = termimantion || truncation;
    (state, reward, done)
}

pub fn epsilon_greedy(mem: &Memory, policy: &dyn Compute, epsilon: f32, obs: &Tensor) -> i64 {
    let mut rng = rand::thread_rng();
    let random_number: f32 = rng.gen::<f32>();
    let action = 
        if random_number > epsilon {
            let value = tch::no_grad(|| policy.forward(&mem, obs));
            let best_action = value.argmax(1, false).int64_value(&[]);
            best_action
        } else {
            sample_discrete_action()
        };
    action
}

pub fn epsilon_update (cur_reward: f32, min_reward: f32, max_reward: f32, min_eps: f32, max_eps: f32) -> f32 {
    if cur_reward < min_reward {
        return max_eps
    }
    let reward_range = max_reward - min_reward;
    let eps_range = max_eps - min_eps;
    let min_update = eps_range / reward_range;
    let new_eps = (max_reward - cur_reward) * min_update;
    if new_eps < min_eps {
        min_eps
    } else {
        new_eps
    }
}

pub struct Transition {
    state: Tensor,
    action: i64,
    reward: f32,
    done: Tensor,
    state_: Tensor,
}

impl Transition {
    pub fn new(state: &Tensor, action: i64, reward: f32, done: bool, state_: &Tensor) -> Self {
        Self {
            state: state.shallow_clone(),
            action: action,
            reward: reward,
            done:  Tensor::from(done as i32 as f32),
            state_: state_.shallow_clone(),
        }
    }
}

pub struct RunningStat <T> {
    values: VecDeque<T>,
    capacity: usize,
}

impl <T> RunningStat <T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::new(),
            capacity: capacity
        }
    }

    pub fn add(&mut self, val: T) 
    {
        self.values.push_back(val);
        if self.values.len() > self.capacity {
            self.values.pop_front();
        }
    }

    pub fn sum(&self) -> T
    where
    T: std::iter::Sum,
    T: Clone,
    {
        self.values.iter().cloned().sum()
    }

    pub fn average(&self) -> f32
    where
    T: std::iter::Sum,
    T: std::ops::Div<f32, Output = T>,
    T: Clone,
    T: Into<f32>,
    {
        let sum = self.sum();
        (sum / (self.capacity as f32)).into()
    }

}

pub struct ReplayMemory {
    transitions: VecDeque<Transition>,
    capacity: usize,
    minsize: usize,
}

impl ReplayMemory {

    pub fn new(capacity: usize, minsize: usize) -> Self{
        Self {
            transitions: VecDeque::new(),
            capacity: capacity,
            minsize: minsize,
        }
    }

    pub fn add(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len()  > self.capacity {
            self.transitions.pop_front();
        }
        
    }

    pub fn sample_batch(&self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<usize> = (0..size)
            .map(|_| rand::thread_rng().gen_range(0..self.transitions.len()))
            .collect();
        let mut states: Vec<Tensor> = Vec::new();
        let mut actions: Vec<i64> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut dones: Vec<Tensor> = Vec::new();
        let mut states_: Vec<Tensor> = Vec::new();
        index.iter().for_each(|i| {
            let transition = self.transitions.get(*i).unwrap();
            states.push(transition.state.shallow_clone());
            actions.push(transition.action);
            rewards.push(transition.reward);
            dones.push(transition.done.shallow_clone());
            states_.push(transition.state_.shallow_clone());
        });   
        (Tensor::stack(&states, 0), 
            Tensor::from_slice(actions.as_slice()).unsqueeze(1), 
            Tensor::from_slice(rewards.as_slice()).unsqueeze(1), 
            Tensor::stack(&dones, 0).unsqueeze(1), 
            Tensor::stack(&states_, 0))     
    }

    pub fn init(&mut self) {
        let mut state = reset();
        let stepskip = 4;
        for s in 0..(self.minsize * stepskip) {
            let action = sample_discrete_action();
            let (state_, reward, done) = step(action);
            if s % stepskip == 0 {
                let t = Transition::new(&state, action, reward, done, &state_);
                self.add(t);
            }
            if done {
                state = reset();
            } else {
                state = state_;
            }
        }
    }

}

