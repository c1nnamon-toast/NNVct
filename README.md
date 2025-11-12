# NNVct
Neural Network Visualizer 

 NNVct can visualize any (supported) NN from specified .onnx file

Example of Abstract Layout
![Abstract Layout of the Network](./images/abstract_layout.png)

Focuced layout for user chosen layer with adjucent layers (2 to the left and 2 to the right)
![Focuces layout for the chosen layer and 2 layers to the left and right](./images/sinco_layout.png)

User can move, hover, pan, zoom etc.
![alt text](./images/hover.png)

User can look at the insides of each* individual neuron
![alt text](./images/inside_of_a_neuron.png)

Currently NNVCt is limited to FFNNs, but the projects stucture allows further modifications and additions

*Input neurons are not supported yet

#### Setup

- install all libraries from requirements.txt
- run ./Application/app.py from root directory
- go to http://127.0.0.1:5000/abstractLayout
