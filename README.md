### Behavioral cloning for autonomous driving using deep ConvNets
ConvNets autonomously drive a vehicle, by controlling steering and velocity.

### Process:
* Collect continuous driving data
* Using Keras design and implement ConvNet  to control driving simulation
* Evaluate performance

---

### Control configurations
Two types of controls have been evaluated:

#### Lateral control
Neural net controls steering angle, while constant set desired speed is modulated according to steering angle.
![LatCtrl](images/Lat_Control_Config.png)

If the vehicle turns hard, we would like to slow down enough to take the turn. Several values for `alpha` were tried. It seems that the value of `0.2` performs satisfactorily. 
Implementation: ```drive.py ```, lines *66-70*.

#### Lateral and longitudinal control
Two deep networks control steering and vehicle set speed, respectively.
![LatLongControlConfig](images/Lat_Long_Config.png)

The speed output of the longitudinal network is clipped in the range `[5mph, 20 mph]`. A speed below `5mph` would stall the vehicle in the simulator. A speed above `20mph` causes severe center-seeking driving behavior in straight segments.

Implementation: ```drive_LLCtrl.py```, lines *66-72*
---

### Network architecture
Codenamed GTRegression

![GTRegression](images/GTRegression.png)

* Initially a quick drop of image size was desired.
* To capture the close correlation, *while* reducing the size as quick as possible, a "large" kernel size (5x5 for the first conv layer) was used, but with a stride of 2 so that local correlation information is retained as much as possible.
* The intermediary layers were selected such that there was a continuous drop in spatial size and an increase in network depth.
* Used spatial dropouts to promote robustness into the extracted feature maps (as opposed to regular random dropout which would yield a reduced learning rate [[link]](https://arxiv.org/pdf/1411.4280.pdf). Spatial dropout causes entire feature maps to be randomly dropped during training, thus forcing other feature maps to be more robust.
* Global average pooling was used before the fully connected layers. Using dropout as a model of creating multiple deep nets, average pooling takes the votes of those nets which specialized in learning more about particular abstractions, as represented by individual feature maps of the last conv layer output. Moreover, if we assume that such underlying factors have been able to fully linearize (disentangle) the latent factors, we can piece-wise approximate the left over non-linearities by a linear combination of these latent representations.
* For the longitudinal controller a relu layer should be used to avoid negative speeds. This however should not happen, as the training set of speeds has only non-negative values. For simplicity, reuse relu accross both controller architectures
* Fully connected layer (128) was used to approximate any non-linearities that were not captured from the earlier layers.
* Fully connected layer (1) which is the measure output (logit).

---



