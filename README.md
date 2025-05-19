# Dream2Real
<h3>Official project repository of Dream2Real</h3>

![](https://github.com/bingozju/Dream2Real/blob/main/visualization/homepageimage.jpg)

Recent advancements in Text-to-3D generation are heavily limited by the capabilities of current 2D vision-language models. When these models attempt to distill complex multi-object descriptions, they often produce 3D outputs that suffer from issues like 3D geometric confusion and the Janus problem. To overcome these challenges, we introduce Dream2Real, a novel framework that views 3D scenes as compositional assemblies of multiple objects. Specifically, Our framework enables the simultaneous optimization of various 3D assets using multi-density neural fields for the first time, which helps maintain a consistent structure and greatly enhances the fidelity of the generated scenes. 
Furthermore, our method reduces the variance in the latent space during the distillation process by decomposing prompts, showing an improved ability to handle abstract textual descriptions and significantly alleviating the Janus problem commonly encountered in Text-to-3D generation.
We provide comprehensive experimental results and visualizations that demonstrate the effectiveness of our proposed method, along with the corresponding theoretical analysis. We believe that this approach has significant potential to contribute the field of 3D generation.

# visualization

### Training Multi-Density Field

M=4, 100epochs, 100 steps for each epoch:

![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/training.gif)


###  Demos of Complex Multi-Object Text-to-3D

*A stuffed **giant panda**, wearing a **cowboy hat**, playing a **cello**, next to a few **bamboo**.* (M=4)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/panda-hat-bamboo-cello.gif)

*A fairytale **cabin** with many **balloons** on the roof, a **mailbox** in front of the door, and a little **fox** sedning a letter.* (M=4)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/cabin-balloons-mailbox-fox.gif)

*A **boy** and a **girl** are sitting around a **tree stump** with a cake on it, with a **tiger** sitting next to them.* (M=5)
![demo](https://github.com/bingozju/Dream2Real/blob/main/visualization/boy-girl-stump-cake-tiger.gif)



# Install

# Usage
```bash
 ./train.sh -w bear-hat-piano-cat -c code/trainfiles/bear-pirate-piano-cat.yaml -g 0;

```


# Acknowledgement

# Citation

