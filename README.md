# Resume_skills


The goal of this project is to build  a model that takes a resume as input and predict the skills of the candidate  and their indicated levels.

The dataset used is from roboflow https://universe.roboflow.com/ons-abderrahim/resume-parser2/dataset/1 and made from around 100 unique resumes. (Many samples are duplicate)

To get labels, we use gpt4o from OpenAI API. This is done in notebooks/Testing OpenAi.ipynb. Each label is a list of pair (Skill, level). where level is in percentage. If there is no skill indicated the level should be 0.

We then train a Donut Model https://huggingface.co/docs/transformers/model_doc/donut. This model has a vision encoder that takes the resume image as input and a text decoder that generates  a text sequence corrresponding to the skills and their associated levels. Training curves and checkpoints can be found here  https://huggingface.co/alakachr/donut-base-resume/tensorboard

We train the model for 3 epochs. Since the dataset a small there is a risk of overfitting , so we could have had  part of the model frozen. But we did not conduct any diagonostic for overfitting since we did not provide  a val dataset during training

Here is an example of model prediction on a  test sample:

![image](https://github.com/user-attachments/assets/f3b40257-8d3d-4079-b303-aeb0055b2bc4)

![image](https://github.com/user-attachments/assets/bbe2dd83-a191-4314-bfb1-3fc7e8da9072)


We evaluated (inside notebooks/Testing_trained_model.ipynb ) the model on small dataset (15) and the model achived 0.25 recall and 0.37 precision on skill name detection. However, for skill level , the model predicted 0 all the time.

Many difficulties can explain those results: dataset is very small, the labels are not accurate (obtained with gpt4) , the class 0 for skill level is over represented, and the model was not properly trained on many epochs (and maybe overfitted a bit).

Requirements for  this project  can be found in pyproject.toml
