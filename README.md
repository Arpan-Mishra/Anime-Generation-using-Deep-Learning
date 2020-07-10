# Anime-Generation-using-Deep-Learning
<b>Built text generation models for genearating Anime synopsis using LSTMs & GPT2</b>

## Table of Contents
  * [Methods Used](#methods-used)
  * [Technologies](#technologies)  
  * [Description](#description)
  * [Data](#data)
  * [Model Building](#model-building)
    * [LSTM](#lstm)
    * [GPT2](#gpt2)
  * [Credits](#credits)
  * [Contact](#contact)
  
  
## Methods Used
* Deep Learning
* Neural Networks
* NLP
* LSTM
* GPT2 

## Technologies
* Python (Spyder, Jupyter Notebook)
* Pandas
* Numpy
* Pytorch
* Hugging Face - Tokenizer, Transformer

## Description
The aim of the project was to see how far technology has come in just a few years when it comes to text generation. <br>
I have used two techniques, LSTM and then a fine tuned GPT2 for generating text and the results are astounding!<br>

## Data
I have generated Anime synopsis, the data was scraped from [myanimelist](imelist.net/anime.php). <br>
The data had initially 16000+ data points but after cleaning and removing some tags we finally have 7300 synopsises.<br>
PS: Anime is basically japanese cartoons. (People who like Anime are going to be really offended by that statement)<br>

## Model Building
### LSTM
* This is a basic text generation model using LSTMs. The input & outputs are both synopsis however the output has been shifted to the right by a single time step.<br>
* The model consists of an Embedding layer, 3 LSTM layers (uni-directional) and 1 linear (or dense) layer.
* I pass the input into the model batch-wise where each batch has a sequence length of 30 tokens.<br> 
* The loss function used here is the cross entropy loss. <br>
* The model output is of the size (batch size * sequence length * vocabulary size).
* During inference we initialize the model with an input text, then the top <i>K</i> most probable words from the outputs
are selected and one word is randomly sampled form them as the next token output (I have used k = 30 here).
 
 Here is a generated Anime synopsis:<br>
<i>'A young woman capable : a neuroi laborer of the human , where one are sent back home ? after defeating everything being their resolve the school who knows if all them make about their abilities . however of those called her past student tar barges together when their mysterious high artist are taken up as planned while to eat to fight ! thus condemning your wish as her lover or being recruited until she regains that of one he goes at work ! with he kills tsuneo with a handsome woman of them or becoming pregnant when haruka discovers with a man ' </i>

Not that great right? The grammar is 70-80% correct but there is no sense in this at all.

### GPT2
* I have used GPT2 with a linear model head from the [Hugging Face](https://huggingface.co/) library for text generation. There are 4 variants of GPT2, I have used GPT2 small which has 117M parameters.<br> 
* For fine tuning the first task is to get the data in the required format, dataloader in Pytorch allows us to that very easily. I appended the `<|endoftext|>` token after every synopsis. 
* The main issue in training was figuring out the batch size and the maximum sequence length so that I don't run out of memory while training on GPU. 
I have used a batch size of 10 and maximum length of 300. 
* The model was trained for 5 epochs. The synopsises with length < max length have been padded using the `<|pad|>` token.
* The model consists of the GPT2 transformer (contains masked self attention and feed forward neural net) and has a language model head, which is similar to the LSTM based 
generator where the inputs and otuputs are same but the outputs are shifted one time step to the right.
* The model outputs a loss (cross entropy loss) and the prediction scores of size (batch size * sequence length * vocabulary size). This score is basically the values before softmax.
* During text generation I have used top-k as well as top-p sampling to generate the next token, doing this is fairly simple using the generate method in hugging face.

Here is a generated Anime synopsis:<br>
<i>In the year 2060, mankind has colonized the solar system, and is now on the verge of colonizing other planets. In order to defend themselves against this new threat, the Earth Federation has established a special unit known as the Planetary Defense Force, or PDF. The unit is composed of the elite Earth Defense Forces, who are tasked with protecting the planet from any alien lifeforms that might threaten the safety of Earth. However, when a mysterious alien ship crashes in the middle of their patrol, they are forced to use their special mobile suits to fend off the alien threat. </i>

This almost sounds like a real Anime!<br>

Here's another example: <br>
<i> A shinigami who is a descendant of the legendary warrior Shigamis father, is sent to Earth to fight against the evil organization known as the Dark Clan. However, his mission is to steal the sacred sword, the Sword of Light, which is said to grant immortality to those who wield it. </i>

Some more [generated anime synopsises](https://github.com/Arpan-Mishra/Anime-Generation-using-Deep-Learning/blob/master/GPT2%20Generator/Generated%20Anime%20Examples.txt) can be found here.

## Credits
* Fine tuning example: https://towardsdatascience.com/teaching-gpt-2-a-sense-of-humor-fine-tuning-large-transformer-models-on-a-single-gpu-in-pytorch-59e8cec40912
* Generating Text using hugging face: https://huggingface.co/blog/how-to-generate

## Contact
* For any queries and feedback please contact me at mishraarpan6@gmail.com

Note: The project is only for education purposes, no plagiarism is intended.



 
 
 
