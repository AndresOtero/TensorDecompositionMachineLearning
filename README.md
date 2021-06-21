# Descomposiciones tensoriales aplicadas a arquitecturas de deep learning
Este repositorio contiene el codigo fuente de mi tesis de grado en Ingenieria Informatica para la Facultad de Ingenieria de la Universidad de Buenos Aires denominada "Descomposiciones tensoriales aplicadas a arquitecturas de Deep Learning".

This repository contains the source code of my degree thesis in Computer Engineering for the Faculty of Engineering of the University of Buenos Aires called "Tensor decompositions applied to Deep Learning architectures".

Author: Andres Otero 

Director: Cesar Caiafa

## Como correr el proyecto

Para poder correr el proyecto es necesario definir ciertos parametros en la clase [NetParams](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/NetParams.py), 
los parametros que siempre deben ser especificados son:

* _model_: alguno de los modelos definidos en [EnumModel](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/EnumModel.py).
* _dataset_ : alguno de los sets de datos definidos en [EnumDataset](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/EnumDataset.py).
* _train_method_ : tiene que ser alguno de los dos metodos definidos en [TrainMethods](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/TrainMethods.py). Las alternativas principales son si el set de datos es de texto o si es de imagenes, esto cambia el proceso de entrenamiento de la red.
* _learning_rate_ : es un parámetro de ajuste que sirve para determinar que tan rapido el modelo se adapta al problema.
* _optimizer_ : cual es el optimizador que utilizara la red, en el trabajo se utilizo principalmente ADAM, para mas detalle ver [OptimizerFactory](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/OptimizerFactory.py).
* _cuda_is_available_ : si la opcion de utilizar [CUDA](https://developer.nvidia.com/cuda-zone) esta disponible en la plataforma donde se realiza el aprendizaje.

Hay varios parametros que tienen valores predeterminados pero se pueden definir en caso de ser necesario:

* _m_ : el tamaño del _feature map_
* _divides_in_row_ : la cantidad de dviisiones en filas del tensor de entrada.
* _divides_in_col_ : la cantidad de divisiones en columnas del tensor de salida.
* _rank_ : el rango elegido del tensor.
* _batch_size_ : el tamaño del lote de entrenamiento.
* _test_batch_size_ : el tamaño del lote para la prueba.
* _epochs_ : la cantidad de epocas de entrenamiento para la red
* _momentum_ : para especificar el momento del optimizador (solo utlizado para el caso de SGD).
* _log_interval_ : el intervalo para el cual la red inserta un mensaje en el log.
* _save_: booleano para especificar si guardar o no el modelo.
* _cuda_ : booleano para especificar si se utiliza o no CUDA.
* _categories_ : cantidad de categorias para las cuales tiene que clasificar la red.
* _log_while_training_ : si loggear mensajes o no durante el entrenamiento
* _tensor_size_ : el tamaño de entrada del tensor.
* _embedding_ : el tamaño del embedding utilizado (en el caso de que este exista).
* _fixed_lenght_ : el tamaño de corte del tensor (0 para que no haya corte).
* _init_value_ : el valor a utilizar para la inicializacion de los tensores (solo utilizdo en el caso de una red tensorial).
* _rank_first_and_last_ : rango del primer y ultimo tensor de la red (solo utilizado en el caso de _Tensor Ring_).
* _dropout_ : dropout utilizado en la red (en el caso que la red lo tenga).

Aqui se agrega un ejemplo de como se define un experimento:
```
result_data = {}
net_params = NetParams(EnumModel.TT_SHARED_MODEL, EnumDataset.FASHION_MNIST_FLAT_DIVISIONS,
              EnumTrainMethods.TRAIN_VISION_METHOD, 1e-4, Constant.ADAM,
              torch.cuda.is_available(), divides_in_col=4,
              divides_in_row=4, tensor_size=Constant.TENSOR_SIZE_MNIST,
              categories=Constant.CATEGORIES_CIFAR, m=64, rank=38,
              init_value=1)
run_net(net_params, result_data)
```

Para mas informacion por favor referirse al  [trabajo](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/src/Tesis_de_Grado.pdf) referenciado.

## How to run an experiment

In order to run ab experiment it is necessary to define certain parameters in the [NetParams] class (https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/NetParams.py),
the parameters that must always be specified are:

* _model_: any of the models defined in [EnumModel] (https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/EnumModel.py).
* _dataset_: any of the data sets defined in [EnumDataset] (https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/EnumDataset.py).
* _train_method_: it has to be one of the two methods defined in [TrainMethods] (https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/TrainMethods.py). The main alternatives are if the data set is text or if it is images, this changes the training process of the network.
* _learning_rate_: it is an adjustment parameter that is used to determine how fast the model adapts to the problem.
* _optimizer_: which is the optimizer that the network will use, ADAM was used mainly at work, for more details see [OptimizerFactory] (https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/Utils/OptimizerFactory.py) .
* _cuda_is_available_: if the option to use [CUDA] (https://developer.nvidia.com/cuda-zone) is available on the platform where the learning takes place.

There are several parameters that have default values ​​but can be defined if necessary:

* _m_: the size of the _feature map_
* _divides_in_row_: the number of divisions in rows of the input tensor.
* _divides_in_col_: the number of divisions in columns of the output tensor.
* _rank_: the chosen range of the tensioner.
* _batch_size_: the size of the training batch.
* _test_batch_size_ - The batch size for the test.
* _epochs_: the number of training epochs for the network
* _momentum_: to specify the moment of the optimizer (only used in the case of SGD).
* _log_interval_: the interval for which the network inserts a message in the log.
* _save_: Boolean to specify whether or not to save the model.
* _cuda_: Boolean to specify whether or not CUDA is used.
* _categories_: number of categories for which you have to classify the network.
* _log_while_training_: whether to log messages or not during training
* _tensor_size_: the input size of the tensor.
* _embedding_: the size of the embedding used (if it exists).
* _fixed_lenght_: the cut size of the tensioner (0 for no cut).
* _init_value_: the value to use for the initialization of the tensors (only used in the case of a tensor network).
* _rank_first_and_last_: rank of the first and last tensor of the network (only used in the case of _Tensor Ring_).
* _dropout_: dropout used in the network (if the network has it).

Here is an example of how an experiment is defined:
```
result_data = {}
net_params = NetParams(EnumModel.TT_SHARED_MODEL, EnumDataset.FASHION_MNIST_FLAT_DIVISIONS,
              EnumTrainMethods.TRAIN_VISION_METHOD, 1e-4, Constant.ADAM,
              torch.cuda.is_available(), divides_in_col=4,
              divides_in_row=4, tensor_size=Constant.TENSOR_SIZE_MNIST,
              categories=Constant.CATEGORIES_CIFAR, m=64, rank=38,
              init_value=1)
run_net(net_params, result_data)
```

For more information please refer to the referenced [paper](https://github.com/AndresOtero/TensorDecompositionMachineLearning/blob/main/src/Tesis_de_Grado.pdf).
