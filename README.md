# Neural-Network Exam

In this repository, we present a new direction for increasing the interpretability of Deep Neural Networks (DNNs) by proposing to replace the linear transforms in DNNs by the **B-cos transform**. 

The B-cos transform is designed to be compatible with existing architectures and we show that it can easily be integrated into common models such as *VGGs*, *ResNets*, *InceptionNets*, and *DenseNets*, whilst maintaining similar performance. 

---

**The B-cos transform**

Typically, the individual neurons in a DNN compute the dot product between their weights **w** and an input **x**:

     f(x; w) = wᵀ x = ||w|| ||x|| c(x, w) with c(x, w) = cos(∠(x, w)).

Here, `∠(x, w)` returns the angle between the vectors **x** and **w**.

In this work, we seek to improve the interpretability of DNNs by promoting weight-input alignment during optimisation. To achieve this, we propose the ***B-cos transform***:

     B-cos(x; w) = ||ŵ|| ||x|| |c(x, ŵ)|ᴮ × sgn (c(x, ŵ)).`

Here, *B* is a hyperparameter, the hat-operator scales **ŵ** to unit norm, and `sgn` denotes the *sign* function. 

Note that this only introduces minor changes with respect to the first equation; e.g., for *B* = 1, the B-cos transform is equivalent to a linear transform with **ŵ**. 

---

**B-cos networks**

The B-cos transform is designed as a *drop-in* replacement of the linear transform, i.e., it can be used in exactly the same way.

For example, a conventional fully connected multi-layer neural network f(**x**; θ) of L layers, is represented by:

      `f(x; θ) = lL ◦ lL−1 ◦ ... ◦ l2 ◦ l1(x),`

with lⱼ denoting layer j with parameters **w**ᵏⱼ for neuron k in layer j, and θ the collection of all model parameters. 

In such a model, each layer lⱼ typically computes:

      `lⱼ(aⱼ; Wⱼ) = φ(Wⱼ aⱼ),`

with aⱼ the input to layer j, φ a non-linear activation function (e.g., ReLU), and the row k of Wⱼ given by the weight vector **w**ᵏⱼ of the k-th neuron in that layer. 

A corresponding **B-cos network** f with layers lⱼ can be formulated in exactly the same way, with the only difference being that every dot product (here between rows of Wⱼ and the input aⱼ) is replaced by the B-cos transform. 

In matrix form, this equates to:

       lⱼ(aⱼ; Wⱼ) = |c(aⱼ; Ŵⱼ)|^(B-1) × (Ŵⱼ aⱼ),`

Here, the power, absolute value, and `×` operators are applied element-wise, `c(aⱼ; Ŵⱼ)` computes the cosine similarity between input aⱼ and the rows of Ŵⱼ, and the hat operator scales the rows of Ŵⱼ to unit norm. 

---
