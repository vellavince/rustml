use rustml::*;
use tch::Tensor;
struct MyModel {
    l1: Linear,
    l2: Linear,
}


impl MyModel {
    fn new (mem: &mut Memory) -> MyModel {
        let l1 = Linear::new(mem, 784, 128);
        let l2 = Linear::new(mem, 128, 10);

        Self {
            l1: l1,
            l2: l2,
        }
    }
}

impl Compute for MyModel {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(mem, input);
        o = o.relu();
        self.l2.forward(mem, &o)
    }
}

fn main() {
    let (x, y) = load_mnist();
    let mut m = Memory::new();
    let mymodel = MyModel::new(&mut m);
    train(&mut m, &x, &y, &mymodel, 100, 128, cross_entropy, 0.001);
    let acc = mymodel.forward(&m, &x).accuracy_for_logits(&y);
    println!("Accuracy: {}", acc);
}

