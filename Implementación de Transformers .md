#### Implementación de Transformers y la visualización de atención

![screen](https://github.com/sandroormeno/Vision-Transformer-ViT/blob/main/images/beans%2028.png)

Para poder valorar la relevancia y usabilidad de estos procedimientos que permiten **visualizar los mecanismos de atención en Transformers**, debemos implementarlos y enfrentarlos a nuestros propios datos. Para ello se nos presenta el siguiente escenario: los modelos basados en Transformers son en principio complejos y requieren gran poder computacional para entrenarlos. Los algoritmos para visualización de mecanismos de atención están diseñados para ciertas arquitecturas. Afortunadamente los modelos que permiten mecanismos de visualización, o que tenemos a nuestro alcance fácilmente, también han sido pre entrenados, lo que nos permite desarrollar transfer learning. En resumen, las pocas arquitecturas disponibles en la actualidad podrán ser reentrenadas para tener una experiencia, en principio, sin una infraestructura potente requerida para entrenar un modelo desde cero.

Los mecanismos de visualización no permiten hacer modificaciones a los modelos de Transformers. Estos deberán ser elegidos cuidadosamente, de entre una variedad de distribuciones. Al ser de arquitectura pública, estos modelos se pueden encontrar en una variedad de distribuciones y con el mismo nombre, esto podría resultar algo confuso. Para el desarrollo de transfer learning, se puede usar diferentes metodologías; pero para nuestro caso, la implementación de visualización de mecanismos de atención, tenemos solo una posibilidad.

Permítame profundizar más en este aspecto. Para desarrollar transfer learning podemos dividir el modelo en *Backbone* y *Head*, ambos pueden ser re entrenados.  En un procedimiento habitual, requerimos solamente entrenar el Head mientras dejamos intacto el Backbone. Para ello podemos incluir un Head personalizado de acuerdo con nuestro requerimiento de clases y la representación. Este procedimiento es el más común y posiblemente obtengamos buenas métricas. Pero, para nuestro experimento, no podemos alterar la arquitectura, afortunadamente el modelo permite reentrenar el Head tan solo modificando la cantidad de clases en la salida; ciertamente con menos representación. De esta manera No hacemos modificación alguna al modelo, requisito fundamental para desarrollar visualización de mecanismos de atención.

``` python
from transformer_explainability_utils.ViT_LRP import deit_tiny_patch16_224 as DeiT_Tiny
from transformers import DeiTFeatureExtractor

feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")
nr_classes = len(labels.names)
IMG_SIZE = feature_extractor.size
URL = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"

model = DeiT_Tiny(pretrained=True, num_classes=nr_classes , input_size=(3, IMG_SIZE, IMG_SIZE), url=URL)

model.to(device)
```

Estas particularidades son mencionadas como desventajas en la investigación publicada de *septiembre del 2022* en [A Survey on Attention Mechanisms for Medical Applications: Are We Moving Toward Better Algorithms?](https://arxiv.org/abs/2204.12406) que he resumido a modo de guía informativa en este post. Esperemos se flexibilicen los procedimientos; pero eso no puede impedir nuestra exploración en este momento, para ello puede encontrar más información y código en este repositorio.

Para dimensionar la profundidad del problema les presento las tres clases que intervienen:  hojas saludables y dos tipos de enfermedad (Angular Leaf Spot y Bean Rust). Puede encontrar más detalles en su [repositorio](https://github.com/AI-Lab-Makerere/ibean/). 

![screen](https://github.com/sandroormeno/Vision-Transformer-ViT/blob/main/images/beans%20class.png)

En el ejemplo podemos apreciar claramente la parte oxidada de la hoja además de la representación de la atención, que ubica con alta precisión la zona afectada. Debemos añadir que el modelo está **87%** seguro que ha realizado una buena clasificación.   En otras ocasiones la atención aparentemente indica una zona dónde claramente no apreciamos deterioro, o no tenemos la sensibilidad para hacer una discriminación asertiva. Un análisis más exhaustivo requiere, además, de otros profesionales, un enfoque *transdisciplinar*. Esto resulta ser otra de las conclusiones que sugiere el estudio mencionado más arriba.

![screen](https://github.com/sandroormeno/Vision-Transformer-ViT/blob/main/images/beans.gif)
