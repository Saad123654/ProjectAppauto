<a name="readme-top"></a>
<br />

<h3 align="center"> Machine Learning class Project </h3>

</div>

# :bento: Structure of the repo

- `configs`
- `data`
- `logs`
- `notebooks`
- `ressources`
- scripts in `src`

# :rocket: Installation

```
git clone https://github.com/Saad123654/ProjectAppauto.git
pip install -r requirements.txt

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
# You can replace with your CUDA version if it is compatible with this version of torch 2.0.1, for instance cu117 for CUDA 11.7 or cu124 for CUDA 12.4
```

# :monocle_face: setup 
install precommit 
```py
pip install pre-commit
pre-commit install
```
then run every commit as it was, a list of test will be run :smile:

#  :whale: Launch an experiment

To modify the hyperparameters as well as the config args: use the adequate config files in `configs/`.

Then in your terminal:
```
python main.py
```

# :green_book: TODO

Add your roadmap here :smile: