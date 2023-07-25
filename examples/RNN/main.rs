use rustml::*;
use tch::Tensor;

struct MyModel {
    l1: RNN,
    l2: Linear,
}

impl MyModel {
    fn new (mem: &mut Memory) -> MyModel {
        let l1 = RNN::new(mem, 1, 10, false, 1,1);
        let l2 = Linear::new(mem, 10, 1);
        Self {
            l1: l1,
            l2: l2,
        }
    }
}

impl Compute for MyModel {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(mem, &input).flat_view();
        o = self.l2.forward(mem, &o);
        o
    }
}

fn main() {
    let (x, y) = generate_random_timeseries(5000,0.5);
    let (input, target) = data_prep_rnn(&y, 11);
    
    let mut m = Memory::new();
    let mymodel = MyModel::new(&mut m);

    train(&mut m, &input, &target, &mymodel, 100, 64, mse, 0.0001);
    tch::no_grad(|| {
        let out = mymodel.forward(&mut m, &input).squeeze();
        graph_compare(&x, &y, &out);    
    });

}

