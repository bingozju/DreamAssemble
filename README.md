# Dream2Real
<h3>Official project repository of Dream2Real</h3>

![](https://github.com/bingozju/Dream2Real/blob/main/homepageimage.jpg)

Recent advancements in Text-to-3D generation are heavily limited by the capabilities of current 2D vision-language models. When these models attempt to distill complex multi-object descriptions, they often produce 3D outputs that suffer from issues like 3D geometric confusion and the Janus problem. To overcome these challenges, we introduce Dream2Real, a novel framework that views 3D scenes as compositional assemblies of multiple objects. Specifically, Our framework enables the simultaneous optimization of various 3D assets using multi-density neural fields for the first time, which helps maintain a consistent structure and greatly enhances the fidelity of the generated scenes. 
Furthermore, our method reduces the variance in the latent space during the distillation process by decomposing prompts, showing an improved ability to handle abstract textual descriptions and significantly alleviating the Janus problem commonly encountered in Text-to-3D generation.
We provide comprehensive experimental results and visualizations that demonstrate the effectiveness of our proposed method, along with the corresponding theoretical analysis. We believe that this approach has significant potential to contribute the field of 3D generation.

<h3>Complex multi-object Text-to-3D</h3>

* If you meet any display problem, please click the corresponding GIF file to view the content properly. This seems to be a problem caused by anonymous, and there is no such issue in github.

***A fairytale cabin with many balloons on the roof, a mailbox in front of the door, and a little fox standing next to the mailbox***
![demo1](https://github.com/bingozju/Dream2Real/blob/main/A%20fairytale%20cabin%20with%20many%20balloons%20on%20the%20roof%2C%20a%20mailbox%20in%20front%20of%20the%20door%2C%20and%20a%20little%20fox%20on%20ground%2C%20sending%20a%20letter.gif)
***A boy and a girl are sitting around a tree stump with a cake on it, with a tiger sitting next to them***
![demo2](https://github.com/bingozju/Dream2Real/blob/main/A%20boy%20and%20a%20girl%20are%20sitting%20around%20a%20tree%20stump%20with%20a%20cake%20on%20it%2C%20with%20a%20tiger%20sitting%20next%20to%20them.gif)


