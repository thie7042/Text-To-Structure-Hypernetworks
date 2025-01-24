# Text-To-Structure-Hypernetworks
![Marching_Cubes](https://github.com/user-attachments/assets/6282bb63-e12f-46d1-ade5-aa74c9443d87)

This repository provides the implementation for a novel text-to-structure hypernetwork model pipeline. The collective output from this method is an implicit neural representation that can adapt to theoretically any representation provided within the training data, allowing the method to be applied to a diverse range of tasks relating to shape, size, schematic and topological form. This interactive framework, coupled with the representation capabilities of implicit neural fields, allows for the potential application of hypernetworks throughout structural engineering and architectural disciplines. 

![Comparison](https://github.com/user-attachments/assets/5cfb4842-442b-424b-bc80-d5d28eca5b41)

The proposed methodology consists of three distinct signal-processing blocks; the transformer module, the hypernetwork and the primary model. The first region of the signal processing takes a tokenised representation of the user’s input, which is passed to the BERT architecture to capture semantic meaning of each token ID within the input.The extracted and normalised design criteria is then processed by a fully connected hypernetwork, tasked with mapping the numerical design specifications into a tensor of weights for the primary network. The primary network generates a neural representation according to the grid of input points, producing the predicted shape of the target structure for the given user input. Notably, the decision boundary between points existing within or outside of this structure defines the topology of the generated solution, such that the resolution of the final design is entirely dependent on the chosen grid size. As the decision boundary is learnt during the training process, it can theoretically take on any desired shape present within the training data given adequate sampling, sufficient model capacity and an appropriate training method. 

![Architecture](https://github.com/user-attachments/assets/a22be8cc-0cbb-403f-94a3-8c8f732ddae3)

The proposed hypernetworks are extremely memory efficient. To demonstrate this, a simple design task has been provided under "example", in which the hypernetwork wights and demo data have been curated to a memory footprint under 30 MB. 
This example also provides a fine-tuning process for BERT-based Named Entity Recognition, to extract the necessary design parameters for the hypernetwork.

The example demo provided within this repository is a simple example of 2D reconstruction, using a base dataset consisting of cantilevered trusses. This has been packaged to allow the entire demo to fit within a single git repository. Applications extend beyond this example, including complex 2D and 3D strucutral design tasks.

![figure](https://github.com/user-attachments/assets/03547ebb-ba23-4675-a228-b870f0c222f4)
