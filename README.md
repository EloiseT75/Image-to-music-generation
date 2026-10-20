\# P***IA***NO — Project Repository



\## Overview



This repository contains the final code submission for the \*\*Multimodal Generative AI course at École Polytechnique\*\*.



The project explores \*\*music generation using Transformer-based models and fine-tuned LLM\*\*.



The repository is organized into \*\*two independent experimental pipelines\*\*, each exploring a different approach to music generation.



Each folder contains its \*\*own implementation, experiments, and documentation\*\*.



Detailed explanations of the architectures, preprocessing pipelines, and design choices are available in the \*\*README file inside each folder\*\*.



\---



\## Repository Structure



\### `Musical\_transformer\_BPE+GPT-like`



This folder contains an implementation of a \*\*Transformer-based music generation model trained on symbolic music tokens\*\*.



The goal of this part of the project is to explore:



\* \*\*music tokenization strategies\*\*

\* \*\*BPE-based vocabulary learning\*\*

\* a \*\*GPT-like autoregressive Transformer architecture\*\*

\* generation of symbolic music sequences



This approach focuses on understanding \*\*how musical representations can be converted into token sequences suitable for Transformer models\*\*.



Additional details about:



\* tokenization

\* model architecture

\* training pipeline

\* generation procedure



are described in the \*\*README inside this folder\*\*.



\---



\### `Visual-to-Music\_Generation\_Mistral7B+CLIP`



This folder contains a \*\*multimodal music generation pipeline\*\* that generates music from \*\*visual inputs\*\*.



The system combines:



\* a \*\*CLIP vision encoder\*\* for extracting image embeddings

\* a \*\*Mistral 7B language model\*\* fine-tuned with \*\*LoRA\*\*

\* \*\*ABC notation generation\*\*

\* conversion of generated music into \*\*MIDI and audio\*\*



The pipeline allows images to influence the generated musical structure.



Further details about:



\* the multimodal architecture

\* dataset preprocessing

\* CLIP embedding integration

\* training configuration

\* music generation pipeline



are described in the \*\*README inside this folder\*\*.



\---



\## Authors



Project developed for the course:



\*\*Multimodal Generative AI — École Polytechnique\*\*



Authors:



\* Eloïse Trabut

\* Antoine Delacour



\---



\## Notes



Each project component is \*\*self-contained\*\* and can be run independently.



For implementation details and instructions on how to run each experiment, please refer to the \*\*README file inside the corresponding folder\*\*.



