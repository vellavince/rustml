use tch::{Tensor};
use rustml::*;

struct MyModel {
    l1: Conv2d,
    l2: Conv2d,
    l3: Linear,
    l4: Linear,
}

impl MyModel {
    fn new (mem: &mut Memory) -> MyModel {
        let l1 = Conv2d::new(mem, 5, 1, 10, 1);
        let l2 = Conv2d::new(mem, 5, 10, 30, 1);
        let l3 = Linear::new(mem, 480, 64);
        let l4 = Linear::new(mem, 64, 10);
        Self {
            l1: l1,
            l2: l2,
            l3: l3,
            l4: l4,
        }
    }
}

impl Compute for MyModel {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(mem, &input);
        o = o.max_pool2d_default(2);
        o = self.l2.forward(mem, &o);
        o = o.max_pool2d_default(2);
        o = o.flat_view();
        o = self.l3.forward(mem, &o);
        o = o.relu();
        o = self.l4.forward(mem, &o);
        o
    }
}

fn main() {
    let (mut x, y) = load_mnist();
    x = x / 250.0;
    x = x.view([-1, 1, 28, 28]);
 
    let mut m = Memory::new();
    let mymodel = MyModel::new(&mut m);    
    train(&mut m, &x, &y, &mymodel, 20, 128, cross_entropy, 0.0001);
    let out = mymodel.forward(&m, &x);
    println!("Accuracy: {}", accuracy(&y, &out));
}
