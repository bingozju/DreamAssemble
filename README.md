# DreamAssemble

# DreamAssemble

**Official implementation of**
**["DreamAssemble: Complex Multi-object Text-to-3D Generation via Multi-Density Neural Fields"](https://github.com/bingozju/DreamAssemble).**

--
Score-distillation methods (DreamFusion, ProlificDreamer, Perp-Neg, …) work well on single objects but tend to break on **complex multi-object prompts**

## Visualization

### Training the Multi-Density Field

Training trajectory with M=4 subspaces, 100 epochs, 100 steps per epoch:

![M=4 training](https://github.com/bingozju/Dream2Real/blob/main/visualization/trainingM4.gif)

Training trajectory with M=6 subspaces, 100 epochs, 100 steps per epoch:

![M=6 training](https://github.com/bingozju/Dream2Real/blob/main/visualization/trainingM6.gif)

### Complex Multi-Object Text-to-3D

*A stuffed **giant panda**, wearing a **cowboy hat**, playing a **cello**, next to a few **bamboo**.* (M=4)

![panda-hat-bamboo-cello](https://github.com/bingozju/Dream2Real/blob/main/visualization/panda-hat-bamboo-cello.gif)

*A fairytale **cabin** with many **balloons** on the roof, a **mailbox** in front of the door, and a little **fox** is holding a letter.* (M=4)

![cabin-balloons-mailbox-fox](https://github.com/bingozju/Dream2Real/blob/main/visualization/cabin-balloons-mailbox-fox.gif)

*A furry **polar bear** wearing a **pirate hat** is playing the **piano**, and **a cat** is sleeping on the piano.* (M=4)

![bear-hat-piano-cat](https://github.com/bingozju/Dream2Real/blob/main/visualization/bear-hat-piano-cat.gif)

*A **boy** and a **girl** are sitting around a **tree stump** with a cake on it, with a **tiger** sitting next to them.* (M=5)

![boy-girl-stump-cake-tiger](https://github.com/bingozju/Dream2Real/blob/main/visualization/boy-girl-stump-cake-tiger.gif)

*A **dragon** is dancing on a **lotus flower**; a little **piglet** and a baby **giraffe** are dancing next to the clown; a little **turtle** is squatting under the leaf; a star-shaped **balloon** is behind them.* (M=6)

![dragon-flower-turtle-piggy-giraffe-balloon](https://github.com/bingozju/Dream2Real/blob/main/visualization/dragon_flower_turtle_piggy_giraffe_balloon.gif)

### Customization

The same scene template with different swapped components — change a single sub-prompt and the rest of the composition is preserved.

*A **dragon** is dancing on a **lotus flower**; a little **piglet** and a baby **giraffe** are dancing next to the clown; a little **turtle** is squatting under the leaf; a star-shaped **balloon** is behind them.* (M=6)

![dragon variant](https://github.com/bingozju/Dream2Real/blob/main/visualization/dragon_flower_turtle_piggy_giraffe_balloon.gif)

*A **clown** is dancing on a **lotus flower**; a little **piglet** and a little **dragon** are dancing next to the clown; a little **turtle** is squatting under the leaf; a star-shaped **balloon** is behind them.* (M=6)

![clown variant](https://github.com/bingozju/Dream2Real/blob/main/visualization/clown_flower_turtle_piggy_dragon_balloon.gif)

### Diversity

Different random seeds and minor sub-prompt variations yield distinct outputs while preserving structural coherence.

| Fox | Bear | Panda |
| :---: | :---: | :---: |
| ![fox 1](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_fox1.gif) | ![bear 1](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_bear1.gif) | ![panda 1](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_panda1.gif) |
| ![fox 2](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_fox2.gif) | ![bear 2](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_bear2.gif) | ![panda 2](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_panda2.gif) |
| ![fox 3](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_fox3.gif) | ![bear 3](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_bear3.gif) | ![panda 3](https://github.com/bingozju/Dream2Real/blob/main/visualization/diversity_panda3.gif) |

### More Text-to-3D

![panda M=3](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-pandaM3.gif)
![bear M=3](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-bearM3.gif)
![robot M=3](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-robotM3.gif)
![students sketch fountain squirrel](https://github.com/bingozju/Dream2Real/blob/main/visualization/students_sketch_fountain_squirrel.gif)
![chess M=6](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-chessingM6.gif)
![animals M=6](https://github.com/bingozju/Dream2Real/blob/main/visualization/single-animalsM6.gif)

## Installation

DreamAssemble requires CUDA, a recent PyTorch (>=2.0), and the same custom
extensions as stable-dreamfusion (`gridencoder`, `freqencoder`, `raymarching`).

```bash
# 1. clone
git clone https://github.com/<your-org>/DreamAssemble.git
cd DreamAssemble

# 2. set up a Python environment
conda create -n dreamassemble python=3.10 -y
conda activate dreamassemble
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuxxx

# 3. project requirements
pip install -r requirements.txt

# 4. install the CUDA extensions from torch-ngp / stable-dreamfusion
#    (clone the upstream repo and run pip install ./gridencoder etc.)
```

### Tetrahedral grids (DMTet stage)

DMTet refinement requires pre-computed tetrahedral grids. Place them in `tets/`:

```
tets/
├── 128_tets.npz
└── 256_tets.npz
```

Grids are available at the
[NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec) repository.

## Usage

The canonical entry point is `train.sh`:

```bash
# Stage 1: NeRF training
./train.sh -w boy_girl_tiger_cake_stump \
           -c trainfiles/boy_girl_tiger_cake_stump.yaml \
           -g 0

# Stage 2: DMTet refinement (loads the stage-1 checkpoint automatically)
./train.sh -w boy_girl_tiger_cake_stump \
           -c trainfiles/boy_girl_tiger_cake_stump.yaml \
           -g 0 -d
```

Flags:

| flag | meaning |
| ---- | ------- |
| `-w` | workspace name (creates `results/<name>/` or `results_dmtet/<name>/`) |
| `-c` | path to a YAML config in `trainfiles/` |
| `-g` | CUDA device id |
| `-d` | enable DMTet refinement (stage 2) |
| `-s` | random seed (default 42) |

Direct invocation is also supported:

```bash
python main.py --config trainfiles/boy_girl_tiger_cake_stump.yaml \
               --workspace results/boy_girl_tiger_cake_stump \
               --gpu 0
```

### Configs

Each YAML in `trainfiles/` describes one scene:

- `text` — global prompt.
- `part_texts` — semicolon-separated sub-prompts (one per subspace).
- `part_centers` — list of 3D centers `mu^j_center` per subspace.
- `part_scales` — radial cropping factor per subspace (used by the edge
  sparsity loss).
- `parts_blob_radius` — initial density-blob radius per subspace.
- camera/sampling parameters: `radius_range`, `theta_range`, `fovy_range`, etc.

Provided examples:

- `boy_girl_tiger_cake_stump.yaml` (M=5)
- `clown_lotus_dance.yaml` (M=6)
- `polar_bear_pirate_piano.yaml` (M=4)
- `panda_cello_bamboos.yaml` (M=4)
- `clown_lego_airplane.yaml` (M=2)

### LLM-assisted config generation

You don't have to write `part_centers` / `part_scales` by hand. The
`llm/` folder ships a small helper that calls an LLM (Qwen by default,
via DashScope's OpenAI-compatible endpoint) and emits a complete YAML in
the same format as the bundled examples. The system prompt — kept in
`llm/system.txt` — tells the model how to decompose a scene into
spatially-coherent subspaces:

```bash
export DASHSCOPE_API_KEY=sk-xxxxx
python llm/llm_gen.py \
    --description "A robot is reading a book under a tree, with a cat at its feet" \
    --output trainfiles/robot_reading_cat.yaml
```

The script writes a self-contained YAML with the LLM-derived `text`,
`part_texts`, `part_centers` and `part_scales`, plus the camera /
rendering defaults shared across the bundled scenes. Review the
generated file (especially the spatial layout) before training; the
existing examples are the canonical reference for what well-tuned
`part_centers` look like, and the LLM is encouraged to emulate them.

You can swap the backend with `--base-url` and `--model`:

```bash
python llm/llm_gen.py \
    --description "..." --output trainfiles/scene.yaml \
    --base-url https://api.openai.com/v1 --model gpt-4o-mini \
    --api-key $OPENAI_API_KEY
```

## Citation
```bibtex
@ARTICLE{11479654,
  author={Huang, Bin and Wang, Jinbao and Jiang, Dongmei and Pei, Hongjuan and Li, Qiulu and Xue, Jian and Lu, Ke},
  journal={IEEE Transactions on Image Processing}, 
  title={DreamAssemble: Complex Multi-Object Text-to-3D Generation via Multi-Density Neural Fields}, 
  year={2026},
  volume={35},
  number={},
  pages={4078-4090},
  keywords={Feeds;Antennas;Electronic mail;Communication systems;Pixel;Protocols;Radio access networks;Regional area networks;Wide area networks;Network architecture;Text-to-3D;distillation sampling;multi-density field;complex multi-object prompts;Janus problem;3D confusion},
  doi={10.1109/TIP.2026.3676627}}
```

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
