# Dream2Real
### Official repository of Dream2Real</h3>

![](https://github.com/bingozju/Dream2Real/blob/main/visualization/homepageimage.jpg)



# Visualization

### Training Multi-Density Field

M=4, 100epochs, 100 steps for each epoch:

![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/trainingM4.gif)

M=6, 100epochs, 100 steps for each epoch:

![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/trainingM6.gif)

###  Demos of Complex Multi-Object Text-to-3D

*A stuffed **giant panda**, wearing a **cowboy hat**, playing a **cello**, next to a few **bamboo**.* (M=4)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/panda-hat-bamboo-cello.gif)

*A fairytale **cabin** with many **balloons** on the roof, a **mailbox** in front of the door, and a little **fox** is holding a letter.* (M=4)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/cabin-balloons-mailbox-fox.gif)

*A furry **polar bear** wearing a **pirate hat** is playing the **piano**, and **a cat** is sleeping on the piano.* (M=4)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/bear-hat-piano-cat.gif)

*A **boy** and a **girl** are sitting around a **tree stump** with a cake on it, with a **tiger** sitting next to them.* (M=5)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/boy-girl-stump-cake-tiger.gif)

*A **dragon** is dancing on a **lotus flower**; a little **piglet** and a baby **giraffe** are dancing next to the clown; a little **turtle** is squatting under the leaf; a star-shaped **balloon** is behind them.(M=6)*
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/dragon_flower_turtle_piggy_giraffe_balloon.gif)

*A **clown** is dancing on a **lotus flower**; a little **piglet** and a little **dragon** are dancing next to the clown; a little **turtle** is squatting under the leaf; a star-shaped **balloon** is behind them.(M=6)*
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/clown_flower_turtle_piggy_dragon_balloon.gif)

###  More Demos
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-pandaM3.gif)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-bearM3.gif)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-robotM3.gif)

![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/single_foxM4.gif)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-chessingM6.gif)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-animalsM6.gif)

# Install

The codebase is built on [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion). For installation, 
```
pip install -r requirements.txt
```

# Usage
```bash

### Use a .yaml config with prompts settings for training.

### bash training
 ./train.sh -w boy-girl-tiger-cake-stump -c trainfiles/boy-girl-tiger-cake-stump-IF.yaml -g 0;
### use Dmtet for refinement of mesh
 ./train.sh -w boy-girl-tiger-cake-stump -c trainfiles/boy-girl-tiger-cake-stump-IF.yaml -g 0 -d;

```


# Acknowledgement

# Citation

