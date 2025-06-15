# Redes-Neuronales-Convolucionales
Se presenta un estudio sobre la aplicación de una arquitectura de red neuronal profunda, ResNet18, para la clasificación de imágenes en seis categorías seleccionadas del dataset Caltech101: "camera", "brain", "Faces", "watch", "ketch", "dollar_bill". El enfoque incluye fine‑tuning, data augmentation y evaluación técnica rigurosa, con el objetivo de medir la capacidad de generalización y la adaptabilidad del modelo en un escenario controlado pero representativo.
![image](https://github.com/user-attachments/assets/d281d817-7978-4f7b-b085-8427f47e0309)
## Justificación técnica
•	Arquitectura ResNet18: Aprovecha bloques residuales para entrenar redes más profundas sin perder la capacidad para propagar gradientes, lo que facilita la convergencia y estabilidad durante el entrenamiento.
•	Transfer learning y fine‑tuning: La adaptación de una red pre entrenada permite reutilizar patrones visuales aprendidos previamente, acelerando el entrenamiento y optimizando el uso de datos disponibles.
•	Data augmentation: La aplicación de transformaciones controladas (rotaciones, volteo horizontal, variaciones de color) busca simular un entorno más diverso, reduciendo la probabilidad de sobreajuste y mejorando el desempeño en imágenes fuera del corpus de entrenamiento.
## Metodología aplicada
1. Selección de categorías: Se eligieron seis clases representativas del dataset Caltech101 para evaluar la capacidad de discriminación del modelo en un escenario de multiclasificación moderada.
2. Preprocesamiento de datos: Las imágenes fueron redimensionadas y sometidas a técnicas de augmentación específicas para enriquecer la variabilidad del dataset de entrenamiento.
3. Adaptación del modelo: La capa final de ResNet18 se reconfiguró para ajustar la salida a seis clases, y se entrenó durante 20 épocas para evaluar la convergencia del modelo y registrar tendencias en la pérdida (loss).
4. Evaluación interna: Se estimaron métricas clave; accuracy, precision, recall, F1-score; además de visualizar la matriz de confusión, con el fin de determinar el rendimiento por clase y nivel de incertidumbre.
5. Evaluación con imágenes externas: Se realizó una prueba con imágenes completamente ajenas al conjunto de entrenamiento, para cuantificar la adaptabilidad del modelo a nuevos contextos visuales.
Este enfoque permite analizar de manera estructurada si una red pre entrenada, con ajuste relativamente breve y técnicas moderadas de enriquecimiento de datos, es capaz de ofrecer resultados robustos y consistentes en tareas específicas de clasificación de imágenes.
## Estructura del código
El código fuente del notebook "Clasificacion_CNN_Caltech101_PyTorch.ipynb" está conformado por las siguientes secciones:
1.	Importación de librerías y configuración del dispositivo
2.	Selección de clases y preparación del dataset
3.	Definición de transformaciones
4.	Creación de DataLoader
5.	Definición y ajuste del modelo CNN
6.	Entrenamiento
7.	Evaluación interna
8.	Evaluación con imágenes externas
El código fuente implementa un flujo completo de aprendizaje supervisado basado en redes neuronales convolucionales, orientado a la clasificación de imágenes pertenecientes a seis categorías seleccionadas del dataset Caltech101. El proceso inicia con la configuración del entorno y la selección de las clases de interés, seguido de una preparación cuidadosa del dataset para asegurar que la estructura sea compatible con los requisitos de una CNN y que la representación de cada clase sea adecuada para el entrenamiento.
Se aplican transformaciones y técnicas de data augmentation sobre las imágenes de entrenamiento con el objetivo de aumentar la variabilidad y robustez del modelo, mientras que para el conjunto de validación se emplean transformaciones simples que preservan la integridad de los datos. La utilización de DataLoader facilita la gestión eficiente de los lotes durante el entrenamiento y la validación, asegurando un flujo de datos continuo y balanceado.
Posteriormente, se adapta y entrena un modelo ResNet18 preentrenado, ajustando su capa final a la cantidad de clases seleccionadas. El proceso de entrenamiento se lleva a cabo durante veinte épocas, permitiendo un aprendizaje progresivo y el monitoreo constante de la pérdida. Una vez finalizado el entrenamiento, el código realiza una evaluación exhaustiva mediante la generación de la matriz de confusión y el cálculo de métricas como precisión, recall y F1-score, lo que proporciona una visión detallada del desempeño del modelo por clase.
Finalmente, el código incorpora una fase de evaluación externa en la que se prueba el modelo con imágenes nuevas no vistas durante el entrenamiento. En esta etapa, se presentan los resultados de predicción junto con las probabilidades y las tres clases más probables para cada imagen. Esta estructura no solo permite validar la capacidad de generalización del modelo, sino que también refleja buenas prácticas en la organización y ejecución de proyectos de aprendizaje profundo, facilitando la interpretación de resultados y la identificación de áreas de mejora.
# Insights principales
## 1. Desempeño del Modelo durante el Entrenamiento
El proceso de aprendizaje se caracterizó por una reducción rápida y sostenida en la función de pérdida, estabilizándose en valores bajos a partir de la segunda mitad del entrenamiento. Este comportamiento evidencia que el modelo fue capaz de captar las características esenciales del conjunto de datos, alcanzando una convergencia efectiva en pocas épocas.
![image](https://github.com/user-attachments/assets/93e70d22-d2f1-4441-83ce-f464e73b5f7a)

Referencia [https://cs231n.github.io/convolutional-networks/]
## 2. Capacidad de Discriminación y Análisis de Errores
El análisis de la matriz de confusión evidencia un rendimiento sobresaliente en la mayoría de las clases, con un patrón predominantemente diagonal que indica alta precisión en la predicción. Las clases "Faces", "camera", "dollar_bill" y "watch" fueron clasificadas correctamente en todos los casos evaluados. Para "brain" y "ketch", se observan leves confusiones: un caso de "ketch" fue confundido con "brain" y otro con "dollar_bill", mientras que la clase "brain" solo presenta una predicción errónea. Este patrón refuerza la solidez del modelo en la mayoría de los escenarios y resalta los puntos donde puede optimizarse aún más.

![image](https://github.com/user-attachments/assets/05eb34f3-dfa8-4e6c-af7b-826fc2c3afa0)

Adicionalmente, el reporte de métricas de clasificación aporta una visión cuantitativa precisa del desempeño del modelo para cada clase, permitiendo identificar fortalezas y áreas de mejora. La siguiente tabla resume los valores de precisión, recall y F1-score para cada categoría, junto con los promedios globales:

![image](https://github.com/user-attachments/assets/86a86ddf-beba-4e76-a5c7-937ee107760f)

Estos resultados reflejan una capacidad notable para identificar correctamente la mayoría de las categorías, así como la necesidad de atención adicional a clases con mayor tendencia a la confusión. La integración de la matriz de confusión y los reportes de clasificación proporciona una base sólida para entender en detalle el rendimiento del modelo y fundamentar futuras estrategias de optimización.
Referencia [https://towardsdatascience.com/how-does-sparse-convolution-work-3257a0a8fd1/]

## 3. Generalización y Robustez ante Imágenes Externas
El desempeño del modelo frente a imágenes no vistas previamente reflejó una tasa de acierto mínima de 47.67%. Los aciertos se concentraron en clases con patrones visuales claros, como el caso de los rostros o los barcos que dentro del grupo escogido representan patrones inequívocos, mientras que las predicciones incorrectas estuvieron asociadas a imágenes con características atípicas o con alta similitud entre clases, como el caso de la imagen externa de un reloj, cuyos patrones de colores pueden asimilarse a las bases de datos usadas en el entrenamiento para identificación de imágenes del cerebro. Esta fase resalta la importancia de la diversidad en el conjunto de datos y el potencial beneficio de estrategias adicionales de data augmentation o fine-tuning para elevar la precisión en contextos externos.

![image](https://github.com/user-attachments/assets/aca1a97c-2902-493d-a9ad-78e4c2e5df85)
![image](https://github.com/user-attachments/assets/2d54874f-0356-4aab-8124-05d19a1d0e6c)
![image](https://github.com/user-attachments/assets/89f5b88a-28be-4f96-bc43-58ab4c366791)
![image](https://github.com/user-attachments/assets/2eaeea78-e974-4d0b-a422-cbe6315376d9)

Estos resultados subrayan la efectividad de la arquitectura ResNet18 ajustada con data augmentation, y destacan la importancia de continuar perfeccionando el pipeline para abordar las situaciones límite observadas tanto en la matriz de confusión como en la evaluación con datos completamente nuevos.
## 4. Estrategias para Mejorar la Precisión del Modelo
Si bien el modelo ha demostrado un desempeño destacado en la tarea de clasificación propuesta, existen diversas estrategias que podrían implementarse para incrementar su precisión y robustez en aplicaciones futuras. En primer lugar, resulta fundamental ampliar la cantidad y diversidad de imágenes disponibles para cada clase. Incorporar ejemplos que reflejen distintas condiciones de iluminación, ángulos y fondos permitiría que la red aprenda representaciones más generales y menos dependientes de características particulares del conjunto original.

![image](https://github.com/user-attachments/assets/03489a43-07ae-42e9-b84f-c980cae753ff)

Asimismo, fortalecer las técnicas de data augmentation podría resultar beneficioso. Más allá de las transformaciones ya aplicadas, es recomendable explorar recortes aleatorios, variaciones más marcadas en el brillo y contraste, e incluso pequeñas distorsiones geométricas. Estas operaciones contribuyen a que el modelo se adapte mejor a la variabilidad inherente a las imágenes en contextos reales y, por ende, minimizan el riesgo de sobreajuste.
Desde la perspectiva del ajuste de la arquitectura y los hiperparámetros, una posible mejora consiste en profundizar el fine-tuning, permitiendo que más capas de la red se actualicen durante el entrenamiento. La utilización de regularización adicional, como dropout o weight decay, puede aportar estabilidad y mejorar la capacidad de generalización. Igualmente, implementar esquemas de validación cruzada ayudaría a obtener una evaluación más representativa del desempeño del modelo y a identificar posibles sesgos vinculados a una partición específica de los datos.
Finalmente, la automatización en la búsqueda de hiperparámetros, mediante técnicas como grid search o random search, junto con un seguimiento sistemático de los experimentos realizados, facilitaría la identificación de configuraciones óptimas para distintos escenarios de clasificación. Aplicar estas mejoras no solo permitiría incrementar la precisión alcanzada, sino también dotar al modelo de una mayor solidez ante nuevos desafíos en el ámbito de la visión por computadora.
[https://keras.io/examples/vision/image_classification_from_scratch/; https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html; https://arxiv.org/abs/1512.03385]




