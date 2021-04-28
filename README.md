
<br><br><br>

# Final Project of ECE590-07: Input Dependent Dynamic CNN Model

## Getting Started
### Train the model
- go to code/main_lab.py and open the function train_val_cifar()
- change the kernel number K: in line 15, change K to 2 ~ 5
- go to code/dynamicConv.py and in line 56, change the temperature to 30
### Reload the model and test the accuracy
- go to code/main_lab.py and open the function report_TSNE()
- note that the pretriained model is located in the file pretrained_model_K{}.pt with different K from 2 to 5
- change the kernel number K and log to False
- open the function test(net) and comment the function log(net)
### Draw the attention map
- go to code/main_lab.py and open the function report_TSNE()
- change the kernel number K and log to True
- open the function log(net) and comment the function test(net)

### Run adversarial attack
- go to code/main_lab.py and open the function adversarial_attack()
