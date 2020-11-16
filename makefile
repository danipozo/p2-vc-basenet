all: experiment-1
	gradient experiments run singlenode \\
		--projectId $(PROJECTID) \\
		--name experiment-1 \\
		--container tensorflow/tensorflow:2.3.1-gpu-jupyter \\
		--machineType P5000 \\
		--command "python experiment-1.py model-files/def-1/ /artifacts/results-1/" \\
		--experimentEnv "{\"EPOCHS\":60,\"PAPERSPACE\":1}" \\
		--workspace "https://github.com/danipozo/p2-vc-basenet.git"
