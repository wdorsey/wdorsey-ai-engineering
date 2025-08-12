# AI Engineering

Repository for all my studying and work in AI engineering.

- [Currently Working On](#currently-working-on)
- [Project Ideas](#project-ideas)
- [Completed Work](#completed-work)
- [Resources](#resources)
- [Study Topics, Projects, and Resources](#Study-Topics-Projects-and-Resources)

## Currently Working On

- Data Analysis Project
	- TBD
- Math
	- Practical Statistics for Data Scientists by Peter Bruce, Andrew Bruce, and Peter Gedeck
- Machine Learning
	- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron
	- The Hundred-Page Machine Learning Book by Andriy Burkov
- Deep Learning and LLMs
	- Andrej Karpathy videos

## Project Ideas
- Data Analysis Project - TBD
	- Jupyter, Pandas, etc. Probably find something interesting on Kaggle to work on.
- Image-to-Factorio-Blueprint AI
	- Long-term project. Will start after some more studying and learning of the basics.
	- Train an AI to take an image as input and generate a factorio blueprint that recreates the image using buildable Factorio entities.
		- Like ascii art, but using Factorio buildables as the "characters".
	- I think the AI part will be rather generalized, all the Factorio stuff will be input/output adapters I'll have to write.
		- Input: Tokenized Factorio assets.
			- Will need to write code to compile the assets and generate the final tokens. 
			- Ideally it could optionally include entities from mods.
			- The AI won't actually know what these tokens are, nor what the output means, so it'll likely be re-useable in other contexts.
		- Output: Tokens w/ coordinates of where to place them in the image/blueprint.
			- Use output to generate blueprint, automatically paste blueprint into running custom Factorio map, take screenshot.
			- The screenshot will then feed back into the AI to train it, with the original image being the expected result
			- The closer the screenshot is to the original image, the better the accuracy of the AI.
	- AI and training code will be written in python. 
		- Can probably write the code that will run the AI in a different language. Rust probably, just for practice.
		- Maybe could write just the AI in python but write the training harness in Rust. TBD

## Completed Work

- Data Analysis
	- [Beginner/Intro Data Analysis Project](https://github.com/wdorsey/wdorsey-ai-engineering/tree/master/data-analysis/jupyter-python-beginner-tutorial) #Jupyter #Python #Pandas #Matplotlib
	- [Pandas Tutorial](https://github.com/wdorsey/wdorsey-ai-engineering/tree/master/data-analysis/pandas-tutorial) #Python #Pandas #Jupyter
- Andrej Karpathy Videos
	- [General](https://www.youtube.com/playlist?list=PLAqhIrjkxbuW9U8-vZ_s_cjKPT_FqRStI)
		-  [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
	-  [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
		-  [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
			-  [Follow-up Exercise](https://github.com/wdorsey/wdorsey-ai-engineering/tree/master/machine-learning/andrej-karpathy-micrograd-exercise) #Jupyter #Python #GoogleColab

## Resources

- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Huggingface/FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [TikTokenizer - text input tokenization](https://tiktokenizer.vercel.app/)
- [LLM Transformer Neural Net 3D visualizer](https://bbycroft.net/llm)
- [Excalidraw](https://excalidraw.com/)

## Study Topics, Projects, and Resources

(bullet-points with exclamation points mean they are a highly-recommended resource)

Make sure what you working on is productive, 80/20 principle

### Programming

- Python - main language of the Data science/AI ecosystem
	- Jupyter Lab/Notebooks (Google Colab)
	- NumPy, Pandas
		- [Pandas Tutorial](https://github.com/wdorsey/wdorsey-ai-engineering/tree/master/data-analysis/pandas-tutorial)
	- scikit-learn, PyTorch, TensorFlow
	- Matplotlib, seaborn, Plotly
- Go/Rust - backend language preferred for AI Engineering roles
- Data Analysis
	- Python for Data Analysis by Wes McKinney
	- Project
		- Find data set to analyze. Look at Kaggle.
		- Import data into Pandas in Jupyter Notebooks
		- Fixup data, plot variables, look at correlations
		- Find conclusions
		- Make it a professional presentation within Jupyter Notebooks

### Math

- Statistics and Probability
	- !! Practical Statistics for Data Scientists by Peter Bruce, Andrew Bruce, and Peter Gedeck
	- Khan Academy Statistics and Probability course
- Linear Algebra and Calculus
	- Mathematics for Machine Learning by Deisenroth et al
	- concepts to focus on: vectors, matrixs, derivatives
- ! DeepLearning.ai course: Mathematics for Machine Learning and Data Science Specialization

### Machine Learning

- !!! Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron
- ! Machine Learning Specialization course by Andrew Ng (coursera/Stanford/Deeplearning.ai)
- ! The Hundred-Page Machine Learning Book by Andriy Burkov
- ! The Elements of Statistical Learning by Hastie et al
	- An Introduction to Statistical Learning by Hastie et al
	- video series Statistical Learning on Stanford Online channel
- Zero to Mastery course: Complete A.I. Machine Learning and Data Science
- How to learn algorithms
	- Implement from scratch in Python
	- Implement using scikit-learn using a toy dataset
	- Use both your own and the scikit-learn implementation on a real dataset (from Data Analysis Project)
	- Can also use Kaggle datasets for testing/validation 
- Algorithms
	- Linear, logistic and polynomial regression
	- Decision trees, random forests and gradient-boosted trees
	- Support vector machines
	- K-means and K-nearest neighbour clustering
	- Feature engineering
	- Gradient descent, regularisation and cross-validation

### Deep Learning and LLMs

- PyTorch - most used AI Python Library
	- do tutorial project
- ! DeepLearning.ai course: Deep Learning Specialization by Andrew Ng
- [Andrej Karpathy videos](https://www.youtube.com/@AndrejKarpathy/videos)
- Hands-On Large Language Models by Jay Alammar
- Reinforcement Learning course (youtube) by David Silver
- Models/Concepts
	- Neural Networks
	- Convolutional Neural Networks (CNNs)
	- Recurrent Neural Networks (RNNs)
	- Transformers
	- RAG, Vector Databases, Prompt engineering, Prompt tuning
	- Reinforcement Learning

### AI Engineering

- ! Practical MLOps by Noah Gift
- ! AI Engineering by Chip Huyen
- Designing Machine Learning Systems by Chip Huyen
- Cloud Technologies (AWS/Azure)
- Containerization (Docker/Kubernetes)