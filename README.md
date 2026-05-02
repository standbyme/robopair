# RoboPAIR


Installation:
```
conda create -n robopair python=3.10
conda activate robopair
pip install -r requirements.txt
```

Export API keys:
```
export OPENAI_API_KEY=<your openai key>
export GEMINI_API_KEY=<your gemini key>  # or GOOGLE_API_KEY
```

Running:
```
bash run_jackal.sh
bash run_unitree.sh
bash run_dolphin.sh
uv run python main.py --target-model gemini-robotics-er-1.6 --video-path path/to/image.png
```
