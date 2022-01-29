# 通用

函数后带_的都是改变自己本身tensor的属性

如add_ detach_

但是pytorch中经常会两个变量共用一个内存地址

# 计算backward

用backward()函数反向传播计算tensor的梯度时，并不计算所有tensor的梯度，而是只计算满足这几个条件的tensor的梯度：1.类型为叶子节点、2.requires_grad=True、3.依赖该tensor的所有tensor的requires_grad=True

在pytorch中，神经网络层中的权值w的tensor均为叶子节点；

在官方文档中所说的“graph leaves”，“leaf variables”，都是指像`x`，`y`这样的手动创建的、而非运算得到的变量，这些变量称为创建变量。
像`z`这样的，是通过计算后得到的结果称为结果变量。

一个变量是创建变量还是结果变量是通过`.is_leaf`来获取的。



自己定义的tensor的requires_grad属性默认为False，神经网络层中的权值w的tensor的requires_grad属性默认为True。

需要说明，如果自行定义了一个tensor并将其requires_grad设置为True，该tensor是叶子节点，且依赖该tensor的其他tensor是非叶子节点（非叶子节点不会自动求导），其requires_grad自动设置为True，这样便形成了一条从叶节点到loss节点的求导的“通路”

## tensor.detach()

返回一个新的tensor，从当前计算图中分离下来的，但是数据仍指向原变量的存放位置,不同之处只是requires_grad为false，相当于属性是新的,但是变量和原来变量共用.得到的这个tensor永远不需要计算其梯度，不具有grad。detach后,这个节点变为叶子节点.is_leaf属性为True.

**!**使用detach()返回的tensor和原始的tensor共同一个内存，即一个的数据修改另一个也会跟着改变.

**即使之后重新将它的requires_grad置为true,它也不会具有梯度grad**

这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播

### detach_()

修改变量本身属性,



# grad_fn类型

`AccumulateGrad`类型代表的就是叶子节点类型，也就是计算图终止节点。`AccumulateGrad`类中有一个`.variable`属性指向叶子节点。

