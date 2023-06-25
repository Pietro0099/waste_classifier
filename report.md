-----------------------------------------------------------------------------------------------------

# BinBrain: The future of Waste Sorting

-----------------------------------------------------------------------------------------------------



## Abstract:

Waste sorting is a critical concern in contemporary society due to the escalating volume of waste being generated, necessitating effective sorting and recycling measures. However, the manual process of waste sorting is hard and time-consuming. To address this challenge, we introduce BinBrain, a microcontroller-powered AI model designed to rapidly and reasonably accurately recognize waste categories for efficient sorting.
BinBrain is characterized by its compact size and remarkable speed, rendering it an ideal solution for waste sorting applications in conjunction with automated industrial machines, as well as public and private environments. Leveraging the capabilities of an artificial neural network model, specifically a Convolutional Neural Network (CNN), BinBrain analyzes images of waste materials to determine their respective types. This approach facilitates efficient waste sorting and recycling procedures.
Our objective is to provide a model suitable for cheap, small, memory-constrained devices in order to make the deployment easy and affordable in every environment.



## 1. Technical Details:


### 1.1 Dataset:
This dataset consists of images from five categories: Biological, Glass, Metal, Paper, and Plastic. The dataset is a combination of multiple online datasets, specifically from Kaggle, and collected samples. The data underwent cleaning processes to remove duplicates, irrelevant images, and correct any labeling errors or inconsistencies. Additionally, a thorough examination of the dataset was conducted to ensure the integrity and quality of the samples. These steps contribute to a reliable and accurate dataset for analysis and application purposes.
The final dataset consist of 14.516 samples, divided as follows(#samples):
Biological: 2955,
Glass: 2980,
Metal: 2801,
Paper: 2969,
Plastic: 2811
As for the Dataset split:
Train: 86.9 %
Validation: 9.7 %
Test: 3.4 %


### 1.2 Model:
The custom CNN model is designed specifically for a particular task,waste sorting, therefore it has been tailored and optimized to address the requirements of the specific problem.
The model incorporates skip connections, also known as residual connections, which allow information to bypass one or more layers in the network. This helps mitigate the vanishing gradient problem, enabling better optimization and more effective learning.
Fused and separable convolutions are utilized in the model. Fused convolutions combine multiple convolutional operations into a single operation, reducing computational complexity. Separable convolutions decompose the standard convolution into separate depthwise and pointwise convolutions, reducing parameters and computational cost.
Fused convolutions, as proposed in [6], are more effective and efficient within the first stages of the network, in fact, the model uses this type of convolutions only in the first third of the architecture.
The model also incorporates Efficient Channel Attention (ECA), proposed in [7], which integrates attention mechanisms to capture channel dependencies explicitly. This technique selectively emphasizes or suppresses features from different channels, enhancing the model's ability to focus on relevant information.
The model adopts the efficient last stage from MobileNetV3, as stated in [5]. MobileNetV3 is a lightweight CNN architecture designed for mobile and embedded devices. By incorporating the efficient last stage, the model reduces computational complexity while still maintaining good performance.
The architecture of the model has undergone careful positioning and optimization of modules and sub-modules. This "surgical" approach implies meticulous design choices in terms of the arrangement, order, and configuration of different components. Such optimization aims to improve the model's performance, efficiency and to meet the restricting memory constraints.


### 1.3 Training and Testing:
The custom CNN model undergoes a comprehensive training process with key elements tailored to optimize its performance.
Firstly, the model is pre-trained on the CIFAR-10 dataset, where images are resized to 128x128 using CAI Super Resolution. This initial training step enables the model to learn general features and representations from a diverse dataset, providing a strong foundation for subsequent fine-tuning.
During training, the chosen loss function is CrossEntropy, which is well-suited for classification tasks. To enhance regularization, label smoothing is applied. This technique helps prevent overconfidence in the model's predictions by smoothing the one-hot encoded target labels, resulting in more calibrated and confident predictions.
The model's optimization is performed using the AdamW optimizer. This variant of the popular Adam optimizer incorporates weight decay as L2 regularization, guarding against overfitting. AdamW's adaptive learning rate mechanism adapts to different datasets, improving the model's ability to converge on optimal solutions.
A custom LR scheduler is implemented to control the learning rate throughout training. It consists of two components: warm-up and ReduceOnPlt. The warm-up phase gradually increases the learning rate at the beginning, allowing the model to explore a wider range of solutions. ReduceOnPlt reduces the learning rate when a specific metric, such as validation loss or accuracy, plateaus. This adjustment fine-tunes the model, aiding in escaping local minima and achieving better performance.
By employing these strategies, the custom CNN model leverages pre-training on CIFAR-10, enhances predictions with label smoothing, optimizes weights with AdamW, and dynamically adjusts the learning rate with a custom LR scheduler. This comprehensive training approach facilitates improved performance, robustness, and generalization on the target task.


### 1.4 Model Compression:
In the post-training phase, the custom CNN model undergoes additional techniques such as quantization and pruning. However, the results of these techniques are as follows:
Quantization: The model is quantized to use 8-bit integer representations instead of the original floating-point precision. Quantization reduces the memory footprint and computational requirements of the model, making it more efficient for deployment on resource-constrained devices. This technique helps optimize the model for execution on hardware with limited capabilities.
Pruning: Unfortunately, the attempt to prune the model did not yield desirable results. Pruning is a technique used to remove redundant or less important weights or connections from the model, thereby reducing its size and computational complexity. However, in this case, the pruning process was unstructured and resulted in unsatisfactory outcomes. It failed due to a combination of hardware limitations and very poor performances of the network.
Given these limitations and challenges, it is likely that the pruning process led to degraded performance or insufficient model compression. It is important to consider the specific constraints and requirements of the hardware and carefully evaluate the impact of pruning on the model's performance before applying this technique.
Therefore, the focus of optimization for this custom CNN model lies in post-training quantization, which successfully reduces the model's memory footprint and computational demands, making it more suitable for deployment on microcontrollers or other resource-constrained devices.


### 1.5 Model Comparison:

A comparison was conducted among four models: EfficientNetV2B0, MobilenetV3 small, Vanilla Model, and BinBrain Model, in terms of their performances.
EfficientNetV2B0 achieved a high test accuracy of 95.59% with 6,052,403 parameters. This model serves as a baseline for comparison due to its relatively large size and strong accuracy performance.
MobilenetV3 small achieved a slightly lower test accuracy of 93.00% with significantly fewer parameters, totaling 506,507. Despite having a smaller model size, MobilenetV3 small maintains a relatively high accuracy compared to EfficientNetV2B0.
The Vanilla Model, which shares the same backbone as the BinBrain Model but lacks skip connections, fused convolutions, and the ECA branch, achieved a test accuracy of 62.59% with only 54,805 parameters. The exclusion of these components appears to have affected its performance, resulting in a lower accuracy compared to EfficientNetV2B0, MobilenetV3 small and, especially, Binbrain.
The BinBrain Model, which incorporates skip connections, fused convolutions, and the ECA branch, achieved a test accuracy of 81.20% with 60,348 parameters. This model strikes a balance between accuracy and model complexity, offering a higher accuracy compared to the Vanilla Model while maintaining a relatively small model size.
In terms of size and compression ratio, EfficientNetV2B0 is the largest model with a size ratio of 1, serving as the baseline. MobilenetV3 small, the Vanilla Model, and the BinBrain Model have significantly smaller sizes, with size ratios of 0.084, 0.009, and 0.010, respectively. This indicates that these models are compressed versions, resulting in much smaller sizes compared to EfficientNetV2B0.
Considering the accuracy ratio, EfficientNetV2B0 achieves an accuracy ratio of 1 as the reference model. MobilenetV3 small, the Vanilla Model, and the BinBrain Model have accuracy ratios of 0.973, 0.655, and 0.849, respectively. These ratios reflect the models' performance relative to EfficientNetV2B0, with MobilenetV3 small performing slightly lower, the Vanilla Model exhibiting significantly lower performance, and the BinBrain Model delivering a respectable accuracy compared to the reference model.
Regarding parameter efficiency, EfficientNetV2B0 is assigned a parameter efficiency of 1. MobilenetV3 small achieves a parameter efficiency of 11.626, indicating higher efficiency compared to EfficientNetV2B0. Binbrain exhibits a parameter efficiency of 85.194, showcasing a much higher efficiency compared to the other models. It also achieved a higher value than the Vanilla Model, with a parameter efficiency of 72.310, showcasing the effectiveness of the optimized sub-modules.
In summary, EfficientNetV2B0 demonstrates high accuracy but has a larger size and fewer parameter efficiency. MobilenetV3 small achieves a slightly lower accuracy but with a significantly smaller model size. The Vanilla Model, lacking skip connections, fused convolutions, and the ECA branch, shows lower accuracy compared to the other models. The BinBrain Model strikes a balance between accuracy, model size, and parameter efficiency, making it a favorable choice among the options provided.



## 2. Applications and Market Perspectives:


### 2.1 Market Overview:
The waste sorting market is growing rapidly, driven by increasing environmental awareness and government regulations. 
This machine learning model can have a good market perspective since it is cheap, versatile, so its deployment would be easy, fast and impactful.
It is thought to improve and optimize already existing sorting mechanisms by easing the human work. We identified two main scenarios where BinBrain can have a positive environmental, economic and social impact. 
In addition a better sorting and collecting chain decreases money loss due to logistics (such as transport, collecting and facilities storing) and recycling of wrongly sorted waste.
These perspectives can explain how the project could be interesting both for private and public funds.


### 2.2 Smart Bin:
Source separation is the most common method in western countries for dealing with the increasing problem of Municipal Solid Waste (MSW). It consists in cleaning and separating waste materials before the collection. Criticality of this technique is that it requires extensive public education since citizens are responsible for separating waste fractions produced in their home or public spaces, but often you experience negligence in this social duty. 
The model can serve as either supervisor or consultant or sorting mechanism for a smart bin deployable  in private and public. In the latter application, it can also assist in monitoring the percentage of waste in urban areas to optimize waste collection efforts.


### 2.3 Recycling Facilities:
Another potential application is in recycling facilities, where the affordable nature of the device makes it suitable as a primary or preliminary tool for indirect sorting chains, hence where sensors detect the presence of materials (not by directly checking their chemical qualities). This helps streamline the operations of more expensive and energy-intensive machinery, enhancing efficiency in recycling processes.



## 3. Ethical Considerations:


### 3.1 Privacy:
One of the most important ethical concerns is privacy. Our model does not collect or share any information, and all data is processed on-device.
This prevents any type of user-profiling in a private environment. The absence of data storage not only protects user privacy but also helps establish trust and confidence in using the AI system.
While our model does not directly share any data, the classification outputs can be a beneficial source of data in public environments for: efficient resource allocation, customized education and awareness campaigns, infrastructure planning, environmental impact assessment.


### 3.2 Fairness:
Training AI models for waste sorting carries the risk of bias within the dataset, which can lead to inaccurate or unfair decisions. This bias may manifest when the training data primarily focuses on specific waste types and logos, potentially resulting in the discrimination against other brands or products. Consequently, users may be influenced to favor certain brands due to the ease of recognition by the waste sorting model.
To address this issue, we took great care in curating our training dataset to ensure diversity and representation of various waste types and materials. By including a wide range of examples, the project aims to minimize any bias that could ‘unfairly’ favor or discriminate against specific brands or products during waste sorting.


### 3.3 Ecological Impact:
The implementation of smart bins presents a trade-off between the potential loss of citizen responsibility in waste sorting and the immediate and positive ecological impact of facilitating waste separation and reducing urban degradation.
For a sustainable future, it is crucial that citizens actively participate in proper waste recycling practices, even with the assistance of technology like smart bins. Informed citizen engagement plays a vital role in minimizing environmental impact and maximizing the effectiveness of waste management efforts.
Additionally, the deployment of smart bins can have significant secondary consequences, including a noticeable improvement in public decorum, leading to a substantial social impact.



## 4. Future Improvements:
BinBrain represents a versatile model for identification, but it carries many criticalities due to its failures in real application scenarios. We want to address some of the main concerns probable in an ideal deployment.


### 4.1 Improvement of the Dataset:
The dataset consists in a wide diversity of waste, but not geographically and environmentally localized. So it could result in a very low accuracy respect to the one resulting from the designing testing. To solve this problem the dataset could be improved locally in single devices: adding photos of waste coming specifically from the geographical location and taken in similar light and environmental conditions of the deployment can reduce the impact of real world deployment.


### 4.2 Multi Label Classifier:
A classifier for waste sorting based only on photos presents big limitations: many common use objects can be produced with different materials, but resulting in the same colors and shape. A suitable example can be bottles: in fact BinBrain has often difficulties in choosing between plastic and glass in this occurrence.
We propose to expand the model by adding other types of data, like weight and volume measures to the classifier. We think that it can lead to decrease cases of uncertainty, since weight/volume ratio for same objects made with different materials (considering only those that the model recognizes) is significantly different.
Conclusion:  
Our microcontroller-powered AI model aims to have a significant ecological impact in public environments by improving waste sorting technology. With its speed, accuracy, and affordability, it can effectively reduce the environmental footprint of waste.
Finally, the model has the potential to enhance decorum and social aspects of public and private urban environments while benefiting companies involved in waste management.












References:
Trends in Solid Waste Management:  https://datatopics.worldbank.org/what-a-waste/trends_in_solid_waste_management.html
Assessing Incorrect Household Waste Sorting in a Medium-Sized Swedish City: https://pdfs.semanticscholar.org/ac1d/d97fdd0ea955ba846f65ae7ea2bbb1014869.pdf
Literature Review of Automated Waste Segregation System using Machine Learning: A Comprehensive Analysis: https://ijssst.info/Vol-20/No-S2/paper15.pdf
A review on automated sorting of source-separated municipal solid waste for recycling: https://www.sciencedirect.com/science/article/pii/S0956053X16305189?casa_token=otwloc9_KtIAAAAA:RCqqO7Tf6YqoLS7AS1F_XuMmD75rE_ajQtiZABMqiW1SujaKude6di1zt6x4g2D_MC1VlZH8kmw
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam: Searching for MobileNetV3, arXiv:1905.02244v5, in 2019.
Mingxing Tan, Quoc V. Le: EfficientNetV2: Smaller Models and Faster Training, arXiv:2104.00298v3, in 2021.
Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, arXiv:1910.03151v4, in 2020.
Pietro Monti, Ludovico Ventura: Waste Classifier, https://github.com/Pietro0099/waste_classifier
