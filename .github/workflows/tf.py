import tensorflow as tf
import matplotlib.pyplot as plt

def loss_function(x):
    return (x - 3) ** 2

x_manual = tf.Variable(initial_value = 5.0, trainable = True, dtype = tf.float32)
learning_rate = 0.1

manual_steps = []
manual_loss = []

for step in range(100):
    with tf.GradientTape() as tape:
        tape.watch(x_manual)
        loss = loss_function(x_manual)
    gradients = tape.gradient(loss, [x_manual])
    x_manual.assign_sub(learning_rate * gradients[0])
    manual_steps.append(step)
    manual_loss.append(loss.numpy())

plt.figure(figsize=(10,6))
plt.plot(manual_steps, manual_loss, marker='o')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Manual Gradient Descent')
plt.show()

x_adam = tf.Variable(initial_value = 5.0, trainable = True, dtype = tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

adam_steps = []
adam_loss = []

for step in range(100):
    with tf.GradientTape() as tape:
        loss = loss_function(x_adam)
    gradients = tape.gradient(loss, [x_adam])
    optimizer.apply_gradients(zip(gradients, [x_adam]))
    adam_steps.append(step)
    adam_loss.append(loss.numpy())

plt.figure(figsize=(10,6))
plt.plot(adam_steps, adam_loss, marker='o')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Adam Optimizer')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(adam_steps, adam_loss, marker='o')
plt.plot(manual_steps, manual_loss, marker='o')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Adam Optimizer')
plt.show()