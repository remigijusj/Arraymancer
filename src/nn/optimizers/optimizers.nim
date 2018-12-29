# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import  ../../tensor/[tensor, higher_order_applymap],
        ../../autograd/autograd,
        ../../private/ast_utils,
        math

# ############################################################
#
#             Generic optimizers
#
# ############################################################

type Optimizer*[TT] = ref object of RootObj
  ## Base class for optimizers
  params*: seq[Variable[TT]] ## Learnable weights

method update*[TT](self: Optimizer[TT]) {.base, inline.} =
  raise newException(ValueError, "base update method is not implemented")

method zeroGrads*[TT](self: Optimizer[TT]) {.base, inline.} =
  # Reset the gradients of the optimized params
  for v in self.params:
    v.grad = v.value.zeros_like

# ############################################################
#
#             SGD: Stochastic Gradient Descent
#
# ############################################################

type
  SGD*[TT] = ref object of Optimizer[TT]
    ## Stochastic gradient descent
    lr*: TT.T # Learning rate. T is the generic parameter of Tensor[T]
    wd*: TT.T # Weight decay

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[Tensor[T]] {.deprecated: "Use the optimizer macro instead".}=
  new(result)
  result = SGD[Tensor[T]](params: @params, lr: learning_rate)

method update*[TT](self: SGD[TT]) =
  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    # v.value -= learning rate * grad
    if v.requires_grad:
      apply2_inline(v.value, v.grad):
        x - self.lr * (y + self.wd * x)
      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

func optimizer*[M, T](model: M, _: typedesc[SGD], learning_rate: T, weight_decay: T = T(0)): SGD[Tensor[T]] =
  ## Create a SGD optimizer that will update the model weight

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  new(result)
  result.params = @[]
  result.lr = learning_rate
  result.wd = weight_decay

  for layer in fields(model):
    when layer is Variable:
      result.params.add layer
    else:
      for field in fields(layer): # TODO recursive for any nesting depth of Model
        when field is Variable:
          result.params.add field

# ############################################################
#
#             Adam: Adaptative Moment Estimation
#
# ############################################################

type
  Adam*[TT] = ref object of Optimizer[TT]
    ## Adaptative Moment Estimation
    learning_rate: TT.T
    weight_decay: TT.T
    beta1, beta2: TT.T              ## Decays on first and second moment
    beta1_t, beta2_t: TT.T          ## Current decay
    first_moments: seq[TT]          ## Exponential moving averages (mean estimation)
    second_moments: seq[TT]         ## Exponential moving averages squared (uncentered variance)
    epsilon: TT.T                   ## Epsilon for numerical stability when dividing

method update*[TT](self: Adam[TT]) =
  # We use the second formulation of Adam from Kingma et al similar to Tensorflow

  # Bias corrected learning rate
  let lr_t = self.learning_rate * sqrt(1 - self.beta2_t) / (1 - self.beta1_t)

  # Raise β1^t and β2^t for next update
  self.beta1_t *= self.beta1
  self.beta2_t *= self.beta2

  for i in 0 ..< self.params.len:
    let v = self.params[i]
    if v.requires_grad:
      # Update biaised first moment estimate
      apply2_inline(self.first_moments[i], v.grad):
        self.beta1 * x + (1 - self.beta1) * y
      # Update biaised second moment estimate
      apply2_inline(self.second_moments[i], v.grad):
        self.beta2 * x + (1 - self.beta2) * y * y
      # Adjust weight
      apply3_inline(v.value, self.first_moments[i], self.second_moments[i]):
        x - lr_t * y / (z.sqrt + self.epsilon) - self.weight_decay * x

      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

func optimizer*[M, T](
        model: M,
        _: typedesc[Adam],
        learning_rate: T = T(0.001),
        beta1: T = T(0.9), beta2: T = T(0.999),
        eps: T = T(1e-8),
        weight_decay: T = T(0)
      ): Adam[Tensor[T]] =
  ## Create a Adam optimizer that will update the model weight

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  new(result)
  result.params = @[]
  result.learning_rate = learning_rate
  result.weight_decay = weight_decay
  result.beta1 = beta1
  result.beta1_t = beta1
  result.beta2 = beta2
  result.beta2_t = beta2
  result.epsilon = eps

  for layer in fields(model):
    when layer is Variable:
      result.params.add layer
      result.first_moments.add layer.grad.zeros_like
      result.second_moments.add layer.grad.zeros_like
    else:
      for field in fields(layer): # TODO recursive for any nesting depth of Model
        when field is Variable:
          result.params.add field
          result.first_moments.add field.grad.zeros_like
          result.second_moments.add field.grad.zeros_like
